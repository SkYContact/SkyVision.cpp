// mcp_server.cpp - Model Context Protocol front end (stdio JSON-RPC 2.0).
//
//   vqa-cli --mcp [--models-dir models] [--tokenizer <path>] [--model <gguf>]
//
// An MCP client (Claude Desktop / CodeBuddy / any agent runtime) launches this
// process and talks newline-delimited JSON-RPC over stdin/stdout, so an agent can
// ask the local GGUF models about images and videos instead of only text.
//
// stdout carries the protocol and nothing else: the real stdout is duplicated to
// a private stream at startup and fd 1 is pointed at stderr, because the
// backends are chatty on stdout ("[perf] ...", token streaming) and one stray
// line would desynchronise the client.
//
// Tools mirror what vqa::Model can do - the same surface the Web UI drives:
// list_models / load_model / describe_image / ask_image / detect_objects /
// point_objects / gaze_point / analyze_video / classify_video.
#include "core/vqa.hpp"

#include "core/text/json.hpp"
#include "core/media/media_io.hpp"
#include "core/runtime/model_registry.hpp"
#include "core/runtime/vqa_spec.hpp"

#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
  #include <io.h>
  #include <process.h>   // _getpid
#else
  #include <unistd.h>
#endif

namespace {

// ---------------------------------------------------------------- protocol I/O
FILE * g_proto = nullptr;   // private duplicate of the original stdout

void proto_write(const std::string & line) {
    if (!g_proto) return;
    fwrite(line.data(), 1, line.size(), g_proto);
    fputc('\n', g_proto);
    fflush(g_proto);
}

// Hand the real stdout to the protocol and redirect fd 1 to stderr, so anything
// the engine prints (perf lines, warnings, streaming pieces) lands in the
// client's log instead of inside a JSON-RPC message.
bool open_protocol_channel() {
#ifdef _WIN32
    const int fd = _dup(_fileno(stdout));
    if (fd < 0) return false;
    g_proto = _fdopen(fd, "wb");
    if (!g_proto) return false;
    _dup2(_fileno(stderr), _fileno(stdout));
#else
    const int fd = dup(fileno(stdout));
    if (fd < 0) return false;
    g_proto = fdopen(fd, "wb");
    if (!g_proto) return false;
    dup2(fileno(stderr), fileno(stdout));
#endif
    setvbuf(g_proto, nullptr, _IOFBF, 1 << 16);
    return true;
}

void mlog(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("[mcp] ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    fflush(stderr);
    va_end(ap);
}

// ---------------------------------------------------------------- server state
vqa::ModelRegistry g_reg;          // one resident model, shared by every tool
std::string g_models_dir = "models";
std::string g_preset_model;        // --model: default when a tool omits "model"
std::string g_live_session;        // owner of the engine-side incremental session

// ---------------------------------------------------------------- media input
// base64 for inline images (data: URI) - an agent often has bytes, not a path
bool base64_decode(const std::string & in, std::string & out) {
    auto val = [](unsigned char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+' || c == '-') return 62;
        if (c == '/' || c == '_') return 63;
        return -1;
    };
    int buf = 0, bits = 0;
    for (unsigned char c : in) {
        if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
        const int v = val(c);
        if (v < 0) return false;
        buf = (buf << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out += (char)((buf >> bits) & 0xFF);
        }
    }
    return !out.empty();
}

bool decode_inline_bytes(const std::string & v, std::string & bytes, std::string & err) {
    size_t at = std::string::npos;
    if (v.rfind("data:", 0) == 0) {
        at = v.find(',');
        if (at == std::string::npos) { err = "malformed data URI (no comma)"; return false; }
        at += 1;
    } else {   // "base64:<payload>"
        at = 7;
    }
    if (!base64_decode(v.substr(at), bytes)) { err = "malformed base64 payload"; return false; }
    return true;
}

bool is_inline(const std::string & v) {
    return v.rfind("data:", 0) == 0 || v.rfind("base64:", 0) == 0;
}

long current_pid() {
#ifdef _WIN32
    return (long)_getpid();
#else
    return (long)getpid();
#endif
}

bool load_image_param(const std::string & v, vqa::MediaInput & media, std::string & err) {
    if (is_inline(v)) {
        std::string bytes;
        if (!decode_inline_bytes(v, bytes, err)) return false;
        if (!vqa::load_image_from_memory(bytes, media) || media.empty()) {
            err = "cannot decode image bytes (png/jpg/bmp expected)";
            return false;
        }
        return true;
    }
    if (!vqa::path_exists(v)) { err = "no such image file: " + v; return false; }
    if (!vqa::load_image(v, media) || media.empty()) {
        err = "cannot decode image: " + v + " (png/jpg/bmp expected)";
        return false;
    }
    return true;
}

bool load_video_param(const std::string & v, double fps, int max_frames, vqa::MediaInput & media,
                      std::string & err) {
    std::string path = v, temp;
    if (is_inline(v)) {
        // ffmpeg only reads files; spool the upload next to the server
        std::string bytes;
        if (!decode_inline_bytes(v, bytes, err)) return false;
        char name[64];
        snprintf(name, sizeof(name), "skyvision_mcp_%ld.tmp.mp4", current_pid());
        temp = name;
        FILE * f = fopen(temp.c_str(), "wb");
        if (!f) { err = "cannot write temporary video file"; return false; }
        fwrite(bytes.data(), 1, bytes.size(), f);
        fclose(f);
        path = temp;
    } else if (!vqa::path_exists(v)) {
        err = "no such video file: " + v;
        return false;
    }
    // a still image is a legal "clip" for every backend (the video towers take
    // it as a single static frame), so a screenshot does not need ffmpeg
    const bool still = path.rfind(".png") != std::string::npos || path.rfind(".PNG") != std::string::npos ||
                       path.rfind(".jpg") != std::string::npos || path.rfind(".JPG") != std::string::npos ||
                       path.rfind(".jpeg") != std::string::npos || path.rfind(".bmp") != std::string::npos ||
                       path.rfind(".webp") != std::string::npos;
    const bool ok = still ? (vqa::load_image(path, media) && !media.empty())
                          : (vqa::load_video_file(path, fps, max_frames, media) && !media.empty());
    if (!temp.empty()) remove(temp.c_str());
    if (!ok) { err = "cannot decode video: " + v + " (ffmpeg required, set FFMPEG)"; return false; }
    return true;
}

// ---------------------------------------------------------------- model slot
// where discovery actually looked (g_models_dir is the raw flag, which may be
// empty when the server defaulted to ./models or .)
const std::string & models_hint() { return g_reg.dir(); }

std::string model_ids() {
    std::string s;
    for (const auto & e : g_reg.entries()) {
        if (!s.empty()) s += ", ";
        s += e.id;
    }
    return s.empty() ? std::string("(none)") : s;
}

// Resolve the "model" argument: an id, a gguf path (registered on the fly), the
// --model preset, or the only runnable model.
const vqa::ModelEntry * resolve_entry(const std::string & want, std::string & err) {
    std::string id = !want.empty() ? want : (!g_preset_model.empty() ? g_preset_model : g_reg.loaded_id());
    if (!id.empty()) {
        if (const vqa::ModelEntry * e = g_reg.find(id)) return e;
        if (vqa::path_exists(id)) {
            if (g_reg.add_path(id, err)) return g_reg.find(id);
            return nullptr;
        }
        err = "unknown model '" + id + "'; runnable ids: " + model_ids();
        return nullptr;
    }
    if (g_reg.entries().size() == 1) return &g_reg.entries()[0];
    if (g_reg.entries().empty()) {
        err = "no runnable model under '" + models_hint() + "' - pass a .gguf path, or start the "
              "server with --models-dir <dir>";
        return nullptr;
    }
    err = "more than one model available, pass model=<id>; runnable ids: " + model_ids();
    return nullptr;
}

vqa::LoadOptions load_opts_from(const vqa::json::Value & a) {
    vqa::LoadOptions o;
    o.n_ctx = a.integer("n_ctx", 0);            // 0 = backend default
    o.n_threads = a.integer("threads", 0);
    o.use_gpu = a.flag("gpu", true);
    o.vision_path = a.str("vision");
    o.region_path = a.str("region");
    return o;
}

// Loads the entry when it is not already resident (one model at a time).
vqa::Model * ensure_model(const vqa::ModelEntry & e, const vqa::json::Value & a, std::string & err) {
    if (g_reg.loaded() && g_reg.loaded_id() == e.id) return g_reg.loaded();
    if (g_reg.loaded_id() != e.id) g_live_session.clear();   // a swap drops the live KV state
    return g_reg.load(e, load_opts_from(a), err);
}

std::string run_ask(vqa::Model * m, const vqa::MediaInput & media, const std::string & prompt,
                    const std::string & mode, const vqa::json::Value & a, std::string & err) {
    vqa::GenParams gp;
    gp.max_tokens = a.integer("max_tokens", 256);
    if (gp.max_tokens <= 0) gp.max_tokens = 256;
    gp.temperature = (float)a.num("temperature", 0.0);
    gp.top_k = a.integer("top_k", 0);
    gp.top_p = (float)a.num("top_p", 1.0);
    gp.stream = false;   // MCP answers in one shot

    const std::string session = a.str("session");
    const bool reset = a.flag("reset", false);
    std::string out;
    if (!session.empty()) {
        // the engine keeps ONE live session: a new name must reset it, or turn 1
        // would splice onto a stranger's KV cache
        if (reset || g_live_session != session) {
            m->session_reset();
            g_live_session = session;
        }
        if (!m->ask_session(media, prompt, gp, out)) {
            // backends put the reason in out_text when they bail out early
            // ("detect needs an image + region model") - surface it verbatim
            err = out.empty() ? std::string("inference failed (see server log)") : out;
            return std::string();
        }
        return out;
    }
    if (reset) {
        m->session_reset();
        g_live_session.clear();
    }
    if (!m->ask(media, prompt, mode, gp, out)) {
        err = out.empty() ? std::string("inference failed (see server log)") : out;
        return std::string();
    }
    return out;
}

// the skill surface comes from the backend's Capabilities; moondream's adapter
// sets the skill flags from which side heads actually loaded.
bool has_skill(const vqa::Capabilities & c, const char * skill) {
    if (!strcmp(skill, "caption")) return c.caption;
    if (!strcmp(skill, "detect")) return c.detect;
    if (!strcmp(skill, "point")) return c.point;
    if (!strcmp(skill, "gaze")) return c.gaze;
    if (!strcmp(skill, "video")) return c.video;
    return c.image;
}

// static per-arch capability spec (vqa_spec.hpp): the "what could this model
// do with the right weights present" answer, as opposed to caps_json's
// "what did the loaded weights actually provide".
vqa::json::Value spec_json(const std::string & arch) {
    auto o = vqa::json::Value::make_object();
    const vqa::ArchSpec * s = vqa::spec_of(arch);
    auto arr = [](const std::vector<std::string> & v) {
        auto a = vqa::json::Value::make_array();
        for (const auto & x : v) a.push(vqa::json::Value::make_string(x));
        return a;
    };
    if (!s) return o;
    std::vector<std::string> tasks, modes;
    for (vqa::Task t : s->tasks) tasks.push_back(vqa::task_name(t));
    for (vqa::Mode m : s->modes) modes.push_back(vqa::mode_name(m));
    o.set("tasks", arr(tasks));
    o.set("modes", arr(modes));
    o.set("inputs", arr(s->inputs));
    if (!s->note.empty()) o.set("note", vqa::json::Value::make_string(s->note));
    return o;
}

vqa::json::Value caps_json(const vqa::ModelEntry & e, const vqa::Capabilities & c) {
    auto o = vqa::json::Value::make_object();
    o.set("image", vqa::json::Value::make_bool(c.image));
    o.set("video", vqa::json::Value::make_bool(has_skill(c, "video")));
    o.set("audio", vqa::json::Value::make_bool(c.audio));
    o.set("caption", vqa::json::Value::make_bool(has_skill(c, "caption")));
    o.set("detect", vqa::json::Value::make_bool(has_skill(c, "detect")));
    o.set("point", vqa::json::Value::make_bool(has_skill(c, "point")));
    o.set("gaze", vqa::json::Value::make_bool(has_skill(c, "gaze")));
    return o;
}

// ---------------------------------------------------------------- small helpers
bool need_str(const vqa::json::Value & a, const char * key, std::string & out, std::string & err) {
    out = a.str(key);
    if (out.empty()) { err = std::string("missing required argument '") + key + "'"; return false; }
    return true;
}

bool need_num(const vqa::json::Value & a, const char * key, double & out, std::string & err) {
    if (!a.has(key)) { err = std::string("missing required argument '") + key + "'"; return false; }
    out = a.num(key, 0.0);
    return true;
}

// moondream's region head answers as "boxes: x1 y1 x2 y2 ..." / "points: x y ...";
// agents want numbers they can index, so the tool layer parses them back out.
bool parse_pairs(const std::string & text, const char * prefix, int dim,
                 std::vector<std::vector<double>> & rows) {
    std::string s = text;
    const size_t plen = strlen(prefix);
    if (s.compare(0, plen, prefix) == 0) s = s.substr(plen);
    std::vector<double> nums;
    const char * p = s.c_str();
    while (*p) {
        while (*p && (isspace((unsigned char)*p) || *p == ',')) p++;
        if (!*p) break;
        char * end = nullptr;
        const double v = strtod(p, &end);
        if (end == p) break;   // trailing prose / "no boxes" - stop at the first junk
        nums.push_back(v);
        p = end;
    }
    rows.clear();
    for (size_t i = 0; i + (size_t)dim <= nums.size(); i += (size_t)dim)
        rows.emplace_back(nums.begin() + (long)i, nums.begin() + (long)i + dim);
    return (int)(nums.size() % (size_t)dim) == 0;
}

// 4 decimals is the region head's own resolution: without rounding the JSON
// fills up with 0.29970000000000002 noise an agent has to wade through
double round4(double v) { return (double)((long long)(v * 10000.0 + (v < 0 ? -0.5 : 0.5))) / 10000.0; }

vqa::json::Value media_json(const vqa::MediaInput & media) {
    auto o = vqa::json::Value::make_object();
    o.set("width", vqa::json::Value::make_number(media.width));
    o.set("height", vqa::json::Value::make_number(media.height));
    o.set("kind", vqa::json::Value::make_string(media.kind == vqa::MediaKind::Video ? "video" : "image"));
    o.set("frames", vqa::json::Value::make_number((double)media.frames.size()));
    return o;
}

// ---------------------------------------------------------------- tools
bool tool_list_models(const vqa::json::Value & a, std::string & out, std::string & err) {
    (void)a;
    (void)err;
    auto arr = vqa::json::Value::make_array();
    for (const auto & e : g_reg.entries()) {
        auto o = vqa::json::Value::make_object();
        o.set("id", vqa::json::Value::make_string(e.id));
        o.set("arch", vqa::json::Value::make_string(e.arch));
        o.set("quant", vqa::json::Value::make_string(e.quant));
        o.set("size_bytes", vqa::json::Value::make_number((double)e.size));
        o.set("path", vqa::json::Value::make_string(e.path));
        o.set("loaded", vqa::json::Value::make_bool(e.id == g_reg.loaded_id()));
        o.set("spec", spec_json(e.arch));
        arr.push(std::move(o));
    }
    auto root = vqa::json::Value::make_object();
    root.set("models_dir", vqa::json::Value::make_string(g_reg.dir()));
    root.set("models", arr);
    root.set("loaded", g_reg.loaded_id().empty() ? vqa::json::Value()
                                                : vqa::json::Value::make_string(g_reg.loaded_id()));
    if (vqa::Model * m = g_reg.loaded())
        if (const vqa::ModelEntry * le = g_reg.find_loaded())
            root.set("caps", caps_json(*le, m->caps()));
    out = vqa::json::dump(root);
    return true;
}

bool tool_load_model(const vqa::json::Value & a, std::string & out, std::string & err) {
    std::string want;
    if (!need_str(a, "model", want, err)) return false;
    const vqa::ModelEntry * e = g_reg.find(want);
    if (!e && vqa::path_exists(want) && g_reg.add_path(want, err)) e = g_reg.find(want);
    if (!e) {
        err = "unknown model '" + want + "'; runnable ids: " + model_ids();
        return false;
    }
    vqa::Model * m = ensure_model(*e, a, err);
    if (!m) return false;
    auto root = vqa::json::Value::make_object();
    root.set("loaded", vqa::json::Value::make_string(e->id));
    root.set("arch", vqa::json::Value::make_string(e->arch));
    root.set("caps", caps_json(*e, m->caps()));
    out = vqa::json::dump(root);
    return true;
}

bool tool_describe_image(const vqa::json::Value & a, std::string & out, std::string & err) {
    std::string image;
    if (!need_str(a, "image", image, err)) return false;
    std::string length = a.str("length", "normal");
    if (length != "short" && length != "normal" && length != "long") {
        err = "length must be one of short / normal / long";
        return false;
    }
    const vqa::ModelEntry * e = resolve_entry(a.str("model"), err);
    if (!e) return false;
    vqa::MediaInput media;
    if (!load_image_param(image, media, err)) return false;
    vqa::Model * m = ensure_model(*e, a, err);
    if (!m) return false;
    if (!m->caps().image) { err = "backend '" + e->arch + "' does not take image input"; return false; }
    // caption is moondream's dedicated skill; otherwise ask for a description
    const bool caption = has_skill(m->caps(), "caption");
    out = run_ask(m, media, caption ? length : std::string("Describe this image in detail."),
                  caption ? std::string("caption") : std::string(), a, err);
    return err.empty();
}

bool tool_ask_image(const vqa::json::Value & a, std::string & out, std::string & err) {
    std::string image, question;
    if (!need_str(a, "image", image, err)) return false;
    if (!need_str(a, "question", question, err)) return false;
    const vqa::ModelEntry * e = resolve_entry(a.str("model"), err);
    if (!e) return false;
    vqa::MediaInput media;
    if (!load_image_param(image, media, err)) return false;
    vqa::Model * m = ensure_model(*e, a, err);
    if (!m) return false;
    if (!m->caps().image) { err = "backend '" + e->arch + "' does not take image input"; return false; }
    out = run_ask(m, media, question, a.str("mode"), a, err);   // mode: "" | ask-think | caption ...
    return err.empty();
}

bool tool_detect_objects(const vqa::json::Value & a, std::string & out, std::string & err) {
    std::string image, target;
    if (!need_str(a, "image", image, err)) return false;
    if (!need_str(a, "target", target, err)) return false;
    const vqa::ModelEntry * e = resolve_entry(a.str("model"), err);
    if (!e) return false;
    vqa::MediaInput media;
    if (!load_image_param(image, media, err)) return false;
    vqa::Model * m = ensure_model(*e, a, err);
    if (!m) return false;
    if (!has_skill(m->caps(), "detect")) {
        err = "backend '" + e->arch + "' has no detect skill (needs moondream3 + *-region.gguf)";
        return false;
    }
    const std::string raw = run_ask(m, media, target, "detect", a, err);
    if (!err.empty()) return false;
    std::vector<std::vector<double>> boxes;
    parse_pairs(raw, "boxes:", 4, boxes);
    auto arr = vqa::json::Value::make_array();
    for (const auto & b : boxes) {
        auto o = vqa::json::Value::make_object();
        o.set("x1", vqa::json::Value::make_number(round4(b[0])));
        o.set("y1", vqa::json::Value::make_number(round4(b[1])));
        o.set("x2", vqa::json::Value::make_number(round4(b[2])));
        o.set("y2", vqa::json::Value::make_number(round4(b[3])));
        if (media.width > 0 && media.height > 0) {
            o.set("x1_px", vqa::json::Value::make_number(round4(b[0] * media.width)));
            o.set("y1_px", vqa::json::Value::make_number(round4(b[1] * media.height)));
            o.set("x2_px", vqa::json::Value::make_number(round4(b[2] * media.width)));
            o.set("y2_px", vqa::json::Value::make_number(round4(b[3] * media.height)));
        }
        arr.push(std::move(o));
    }
    auto root = vqa::json::Value::make_object();
    root.set("target", vqa::json::Value::make_string(target));
    root.set("count", vqa::json::Value::make_number((double)boxes.size()));
    root.set("boxes", arr);
    root.set("media", media_json(media));
    root.set("note", vqa::json::Value::make_string(
        "coordinates are normalised to the media size (0..1); *_px are the same values in pixels"));
    if (boxes.empty()) root.set("raw", vqa::json::Value::make_string(raw));
    out = vqa::json::dump(root);
    return true;
}

bool tool_point_objects(const vqa::json::Value & a, std::string & out, std::string & err) {
    std::string image, target;
    if (!need_str(a, "image", image, err)) return false;
    if (!need_str(a, "target", target, err)) return false;
    const vqa::ModelEntry * e = resolve_entry(a.str("model"), err);
    if (!e) return false;
    vqa::MediaInput media;
    if (!load_image_param(image, media, err)) return false;
    vqa::Model * m = ensure_model(*e, a, err);
    if (!m) return false;
    if (!has_skill(m->caps(), "point")) {
        err = "backend '" + e->arch + "' has no point skill (needs moondream3 + *-region.gguf)";
        return false;
    }
    const std::string raw = run_ask(m, media, target, "point", a, err);
    if (!err.empty()) return false;
    std::vector<std::vector<double>> pts;
    parse_pairs(raw, "points:", 2, pts);
    auto arr = vqa::json::Value::make_array();
    for (const auto & p : pts) {
        auto o = vqa::json::Value::make_object();
        o.set("x", vqa::json::Value::make_number(round4(p[0])));
        o.set("y", vqa::json::Value::make_number(round4(p[1])));
        if (media.width > 0 && media.height > 0) {
            o.set("x_px", vqa::json::Value::make_number(round4(p[0] * media.width)));
            o.set("y_px", vqa::json::Value::make_number(round4(p[1] * media.height)));
        }
        arr.push(std::move(o));
    }
    auto root = vqa::json::Value::make_object();
    root.set("target", vqa::json::Value::make_string(target));
    root.set("count", vqa::json::Value::make_number((double)pts.size()));
    root.set("points", arr);
    root.set("media", media_json(media));
    root.set("note", vqa::json::Value::make_string("normalised 0..1 coordinates; *_px are pixels"));
    if (pts.empty()) root.set("raw", vqa::json::Value::make_string(raw));
    out = vqa::json::dump(root);
    return true;
}

bool tool_gaze_point(const vqa::json::Value & a, std::string & out, std::string & err) {
    std::string image;
    if (!need_str(a, "image", image, err)) return false;
    double x, y;
    if (!need_num(a, "x", x, err)) return false;
    if (!need_num(a, "y", y, err)) return false;
    if (x < 0 || x > 1 || y < 0 || y > 1) {
        err = "x / y must be normalised coordinates in 0..1";
        return false;
    }
    const vqa::ModelEntry * e = resolve_entry(a.str("model"), err);
    if (!e) return false;
    vqa::MediaInput media;
    if (!load_image_param(image, media, err)) return false;
    vqa::Model * m = ensure_model(*e, a, err);
    if (!m) return false;
    if (!has_skill(m->caps(), "gaze")) {
        err = "backend '" + e->arch + "' has no gaze skill (needs moondream3 + *-region.gguf)";
        return false;
    }
    char prompt[64];
    snprintf(prompt, sizeof(prompt), "%.4f,%.4f", x, y);
    const std::string raw = run_ask(m, media, prompt, "gaze", a, err);
    if (!err.empty()) return false;
    std::vector<std::vector<double>> g;
    parse_pairs(raw, "gaze:", 2, g);
    auto from = vqa::json::Value::make_object();
    from.set("x", vqa::json::Value::make_number(round4(x)));
    from.set("y", vqa::json::Value::make_number(round4(y)));
    auto root = vqa::json::Value::make_object();
    root.set("from", std::move(from));
    if (!g.empty()) {
        auto o = vqa::json::Value::make_object();
        o.set("x", vqa::json::Value::make_number(round4(g[0][0])));
        o.set("y", vqa::json::Value::make_number(round4(g[0][1])));
        if (media.width > 0 && media.height > 0) {
            o.set("x_px", vqa::json::Value::make_number(round4(g[0][0] * media.width)));
            o.set("y_px", vqa::json::Value::make_number(round4(g[0][1] * media.height)));
        }
        root.set("gaze", o);
    } else {
        root.set("gaze", vqa::json::Value());
        root.set("raw", vqa::json::Value::make_string(raw));
    }
    root.set("media", media_json(media));
    out = vqa::json::dump(root);
    return true;
}

bool tool_analyze_video(const vqa::json::Value & a, std::string & out, std::string & err) {
    std::string video;
    if (!need_str(a, "video", video, err)) return false;
    const double fps = a.num("fps", 2.0);
    const int max_frames = a.integer("max_frames", 32);
    const vqa::ModelEntry * e = resolve_entry(a.str("model"), err);
    if (!e) return false;
    vqa::MediaInput media;
    if (!load_video_param(video, fps, max_frames, media, err)) return false;
    vqa::Model * m = ensure_model(*e, a, err);
    if (!m) return false;
    if (!has_skill(m->caps(), "video")) {
        err = "backend '" + e->arch + "' does not take video input (use VideoChat3 / Qwen3.5 / "
              "Marlin / MOSS-VL / VideoMAE)";
        return false;
    }
    std::string question = a.str("question");
    std::string mode = a.str("mode");
    const bool tower = vqa::arch_info(e->arch).pure_vision;
    if (tower && mode.empty()) mode = "5";   // label distribution instead of a caption
    if (question.empty() && !tower) question = "Describe what happens in this video.";
    out = run_ask(m, media, question, mode, a, err);
    return err.empty();
}

bool tool_classify_video(const vqa::json::Value & a, std::string & out, std::string & err) {
    std::string video;
    if (!need_str(a, "video", video, err)) return false;
    const double fps = a.num("fps", 2.0);
    const int max_frames = a.integer("max_frames", 16);
    const vqa::ModelEntry * e = resolve_entry(a.str("model"), err);
    if (!e) return false;
    if (!vqa::arch_info(e->arch).pure_vision) {
        err = "classify_video needs a video tower (videomae / vivit); '" + e->arch +
              "' is a language backend - use analyze_video instead";
        return false;
    }
    std::string mode = "embed";
    if (!a.flag("embed", false)) {
        const int k = a.integer("top_k", 5);
        mode = std::to_string(k > 0 ? k : 5);
    }
    vqa::MediaInput media;
    if (!load_video_param(video, fps, max_frames, media, err)) return false;
    vqa::Model * m = ensure_model(*e, a, err);
    if (!m) return false;
    out = run_ask(m, media, std::string(), mode, a, err);
    return err.empty();
}

bool tool_classify_image(const vqa::json::Value & a, std::string & out, std::string & err) {
    std::string image;
    if (!need_str(a, "image", image, err)) return false;
    const vqa::ModelEntry * e = resolve_entry(a.str("model"), err);
    if (!e) return false;
    if (!vqa::arch_info(e->arch).pure_vision) {
        err = "classify_image needs an image tower (siglip / videomae / vivit); '" + e->arch +
              "' is a language backend - use ask_image instead";
        return false;
    }
    std::string mode;
    if (!a.flag("embed", false)) {
        const int k = a.integer("top_k", 5);
        mode = std::to_string(k > 0 ? k : 5);   // tower top-k: siglip labels / videomae labels
    }   // else: siglip prints the pooled-embedding summary
    vqa::MediaInput media;
    if (!load_image_param(image, media, err)) return false;
    vqa::Model * m = ensure_model(*e, a, err);
    if (!m) return false;
    out = run_ask(m, media, std::string(), mode, a, err);
    return err.empty();
}

bool tool_embed_image(const vqa::json::Value & a, std::string & out, std::string & err) {
    std::string image;
    if (!need_str(a, "image", image, err)) return false;
    const vqa::ModelEntry * e = resolve_entry(a.str("model"), err);
    if (!e) return false;
    vqa::MediaInput media;
    if (!load_image_param(image, media, err)) return false;
    vqa::Model * m = ensure_model(*e, a, err);
    if (!m) return false;
    vqa::EmbedParams ep;
    ep.truncate_dim = a.integer("dim", 0);
    std::vector<float> vec;
    if (!m->embed(media, std::string(), ep, vec)) {
        err = "backend '" + e->arch + "' has no embedding path (embed_image needs "
              "embgemma2 / siglip)";
        return false;
    }
    // one L2-normalized vector: cosine against other embed_image outputs ranks
    // similarity (scripts/siglip_retrieve.py shows the pattern end to end)
    char buf[48];
    out = "{\"dim\": " + std::to_string(vec.size()) + ", \"normalized\": true, \"embedding\": [";
    for (size_t i = 0; i < vec.size(); i++) {
        snprintf(buf, sizeof(buf), "%s%.6f", i ? ", " : "", (double)vec[i]);
        out += buf;
    }
    out += "]}";
    return true;
}

using ToolFn = bool (*)(const vqa::json::Value &, std::string &, std::string &);

struct ToolDef {
    const char * name;
    ToolFn fn;
};

const ToolDef kTools[] = {
    { "list_models", tool_list_models },
    { "load_model", tool_load_model },
    { "describe_image", tool_describe_image },
    { "ask_image", tool_ask_image },
    { "detect_objects", tool_detect_objects },
    { "point_objects", tool_point_objects },
    { "gaze_point", tool_gaze_point },
    { "analyze_video", tool_analyze_video },
    { "classify_video", tool_classify_video },
    { "classify_image", tool_classify_image },
    { "embed_image", tool_embed_image },
};

// Tool declarations. Schemas are the contract an agent sees, so the
// descriptions state the constraints (which backend supports what) instead of
// leaving them to be discovered by failing calls.
const char * TOOLS_JSON = R"MCPJSON([
  {
    "name": "list_models",
    "description": "List the GGUF models this SkyVision build can run from its models directory, with architecture, quantisation, size and which one is currently resident in memory. Also reports the loaded model's capabilities (image/video/caption/detect/point/gaze). Call this first when you do not know a model id.",
    "inputSchema": { "type": "object", "properties": {} }
  },
  {
    "name": "load_model",
    "description": "Load a vision-language model into memory. Only one model is resident at a time, so loading another unloads the current one; most tools load implicitly, so call this only to pin a model up front or to set context length / thread count.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "model": { "type": "string", "description": "Model id from list_models, or the path of a .gguf file." },
        "n_ctx": { "type": "integer", "description": "KV context window in tokens (0 = backend default)." },
        "threads": { "type": "integer", "description": "CPU threads (0 = auto)." },
        "gpu": { "type": "boolean", "description": "Prefer the CUDA backend when the build has one (default true)." },
        "vision": { "type": "string", "description": "moondream3 only: explicit *-vision.gguf path." },
        "region": { "type": "string", "description": "moondream3 only: explicit *-region.gguf path (detect/point/gaze)." }
      },
      "required": ["model"]
    }
  },
  {
    "name": "describe_image",
    "description": "Caption an image (short sentence / normal / detailed paragraph). Fastest way to let a text-only model 'see' a file.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "image": { "type": "string", "description": "Image to look at: a local file path (png/jpg/bmp), or inline bytes as data:image/...;base64,<data> or base64:<data>." },
        "length": { "type": "string", "enum": ["short", "normal", "long"], "description": "Caption length (default normal)." },
        "max_tokens": { "type": "integer", "description": "Answer budget (default 256)." },
        "model": { "type": "string", "description": "Model id or .gguf path; omit when only one model is available." }
      },
      "required": ["image"]
    }
  },
  {
    "name": "ask_image",
    "description": "Ask a free-form question about an image: object/scene recognition, reading text, counting, comparing, spatial relations. Use describe_image instead when you just need a caption.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "image": { "type": "string", "description": "Local image path, or inline data:image/...;base64,<data> / base64:<data>." },
        "question": { "type": "string", "description": "Question about the image, in any language." },
        "mode": { "type": "string", "description": "Optional skill override: 'ask-think' enables moondream grounded reasoning (slower, with coordinates), 'caption' captions. Leave empty for a plain answer." },
        "max_tokens": { "type": "integer", "description": "Answer budget (default 256)." },
        "temperature": { "type": "number", "description": "Sampling temperature, 0 = greedy (default 0)." },
        "top_k": { "type": "integer", "description": "top-k sampling (0 = disabled)." },
        "top_p": { "type": "number", "description": "nucleus sampling." },
        "session": { "type": "string", "description": "Multi-turn session id: later calls with the same id keep the model's engine-side context (MOSS-VL / realtime)." },
        "reset": { "type": "boolean", "description": "Drop the session's context before this turn." },
        "model": { "type": "string", "description": "Model id or .gguf path." }
      },
      "required": ["image", "question"]
    }
  },
  {
    "name": "detect_objects",
    "description": "Detect every instance of a target in an image and return bounding boxes as JSON (normalised 0..1 plus pixel coordinates). Requires a moondream3 model with its *-region.gguf head.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "image": { "type": "string", "description": "Local image path, or inline base64 / data: URI." },
        "target": { "type": "string", "description": "Object to find, e.g. 'person', 'traffic light', '猫'." },
        "model": { "type": "string", "description": "Model id or .gguf path." }
      },
      "required": ["image", "target"]
    }
  },
  {
    "name": "point_objects",
    "description": "Locate the centre of a target in an image as point coordinates (JSON, normalised + pixels) - cheaper than detect when you only need one representative position. Requires moondream3 + *-region.gguf.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "image": { "type": "string", "description": "Local image path, or inline base64 / data: URI." },
        "target": { "type": "string", "description": "Object to point at, e.g. 'the dog'." },
        "model": { "type": "string", "description": "Model id or .gguf path." }
      },
      "required": ["image", "target"]
    }
  },
  {
    "name": "gaze_point",
    "description": "Predict where the person at a given start point is looking, as a target point in the image (JSON, normalised + pixels). Requires moondream3 + *-region.gguf.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "image": { "type": "string", "description": "Local image path, or inline base64 / data: URI." },
        "x": { "type": "number", "description": "Start point x, normalised 0..1 (the person's eye/head)." },
        "y": { "type": "number", "description": "Start point y, normalised 0..1." },
        "model": { "type": "string", "description": "Model id or .gguf path." }
      },
      "required": ["image", "x", "y"]
    }
  },
  {
    "name": "analyze_video",
    "description": "Ask a question about a video (or a frame sequence): what happens, in what order, at which timestamps. Frames are sampled with ffmpeg (fps / max_frames). Video towers (VideoMAE / ViViT) answer with action labels instead of prose - use classify_video for those.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "video": { "type": "string", "description": "Local video path (mp4/mkv/...), or inline base64 / data: URI (spooled to a temp file)." },
        "question": { "type": "string", "description": "Question about the clip, e.g. 'What happens and when?'." },
        "fps": { "type": "number", "description": "Sampling rate for frame extraction (default 2)." },
        "max_frames": { "type": "integer", "description": "Frame cap; the clip is resampled uniformly when it binds (default 32)." },
        "mode": { "type": "string", "description": "Backend skill override: Marlin-2B supports 'caption' (scene + timestamped events) and 'find' (locate an event in time); 'ask-think' for grounded reasoning." },
        "max_tokens": { "type": "integer", "description": "Answer budget (default 256)." },
        "session": { "type": "string", "description": "Multi-turn / realtime session id (MOSS-VL)." },
        "reset": { "type": "boolean", "description": "Drop the session's context before this turn." },
        "model": { "type": "string", "description": "Model id or .gguf path." }
      },
      "required": ["video"]
    }
  },
  {
    "name": "classify_video",
    "description": "Run a pure video tower (VideoMAE / ViViT, Kinetics-400) over a clip: top-k action labels with probabilities, or the pooled embedding vector when embed=true. No text prompt is involved.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "video": { "type": "string", "description": "Local video path, or inline base64 / data: URI." },
        "top_k": { "type": "integer", "description": "How many labels to return (default 5)." },
        "embed": { "type": "boolean", "description": "Return the raw pooled embedding instead of labels." },
        "fps": { "type": "number", "description": "Sampling rate for frame extraction (default 2)." },
        "max_frames": { "type": "integer", "description": "Frame cap (default 16; the tower resamples to its own clip length)." },
        "model": { "type": "string", "description": "Model id or .gguf path." }
      },
      "required": ["video"]
    }
  },
  {
    "name": "classify_image",
    "description": "Run a pure image tower over a still image: top-k class labels with probabilities, or the pooled-embedding summary when embed=true. Needs an image-tower checkpoint - siglip (e.g. a SigLIP NSFW/label fine-tune) or videomae/vivit. No text prompt is involved; for descriptions use describe_image / ask_image.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "image": { "type": "string", "description": "Local image path, or inline base64 / data: URI." },
        "top_k": { "type": "integer", "description": "How many labels to return (default 5)." },
        "embed": { "type": "boolean", "description": "Return the pooled-embedding summary instead of labels." },
        "model": { "type": "string", "description": "Model id or .gguf path." }
      },
      "required": ["image"]
    }
  },
  {
    "name": "embed_image",
    "description": "Embed a still image into one L2-normalized vector (embgemma2 multimodal or siglip image tower). Cosine similarity between two embed_image outputs ranks image similarity - build galleries with one call per image and compare, or follow scripts/siglip_retrieve.py.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "image": { "type": "string", "description": "Local image path, or inline base64 / data: URI." },
        "dim": { "type": "integer", "description": "Matryoshka truncation for encoders that support it (0 = model default)." },
        "model": { "type": "string", "description": "Model id or .gguf path." }
      },
      "required": ["image"]
    }
  }
])MCPJSON";

// The schema above is written pretty-printed for humans, but MCP frames one
// message per line: a literal newline inside the response would desynchronise a
// client that reads line-wise. Parse it once, re-dump it compact (which also
// validates the schema at runtime).
const std::string & tools_schema_compact() {
    static std::string cached;
    if (cached.empty()) {
        vqa::json::Value v;
        std::string err;
        if (vqa::json::parse(TOOLS_JSON, v, err)) {
            cached = vqa::json::dump(v);
        } else {
            mlog("tool schema is not valid JSON (%s) - serving it verbatim", err.c_str());
            cached = TOOLS_JSON;
        }
    }
    return cached;
}

// ---------------------------------------------------------------- JSON-RPC
void reply_result(const vqa::json::Value & id, const std::string & result_json) {
    std::string msg = "{\"jsonrpc\":\"2.0\",\"id\":";
    msg += id.is_null() ? std::string("null") : vqa::json::dump(id);
    msg += ",\"result\":";
    msg += result_json;
    msg += "}";
    proto_write(msg);
}

void reply_error(const vqa::json::Value & id, int code, const std::string & message) {
    auto e = vqa::json::Value::make_object();
    e.set("code", vqa::json::Value::make_number(code));
    e.set("message", vqa::json::Value::make_string(message));
    std::string msg = "{\"jsonrpc\":\"2.0\",\"id\":";
    msg += id.is_null() ? std::string("null") : vqa::json::dump(id);
    msg += ",\"error\":";
    msg += vqa::json::dump(e);
    msg += "}";
    proto_write(msg);
}

// Tool failures are results with isError, not JSON-RPC errors (spec: the client
// shows the text to the model so it can retry with corrected arguments).
void reply_tool(const vqa::json::Value & id, const std::string & text, bool is_error) {
    auto block = vqa::json::Value::make_object();
    block.set("type", vqa::json::Value::make_string("text"));
    block.set("text", vqa::json::Value::make_string(text));
    auto content = vqa::json::Value::make_array();
    content.push(std::move(block));
    auto result = vqa::json::Value::make_object();
    result.set("content", content);
    result.set("isError", vqa::json::Value::make_bool(is_error));
    reply_result(id, vqa::json::dump(result));
}

const char * kProtocolVersions[] = { "2024-11-05", "2025-03-26", "2025-06-18" };

std::string negotiate_version(const std::string & asked) {
    for (const char * v : kProtocolVersions)
        if (asked == v) return asked;
    return kProtocolVersions[sizeof(kProtocolVersions) / sizeof(kProtocolVersions[0]) - 1];
}

void handle_initialize(const vqa::json::Value & id, const vqa::json::Value & params) {
    auto result = vqa::json::Value::make_object();
    result.set("protocolVersion", vqa::json::Value::make_string(negotiate_version(params.str("protocolVersion"))));
    auto tools = vqa::json::Value::make_object();
    tools.set("listChanged", vqa::json::Value::make_bool(false));
    auto caps = vqa::json::Value::make_object();
    caps.set("tools", tools);
    result.set("capabilities", caps);
    auto info = vqa::json::Value::make_object();
    info.set("name", vqa::json::Value::make_string("skyvision"));
    info.set("version", vqa::json::Value::make_string("1.0.0"));
    result.set("serverInfo", info);
    result.set("instructions", vqa::json::Value::make_string(
        "SkyVision.cpp serves local GGUF vision-language models. Call list_models once to see what "
        "is runnable, then describe_image / ask_image for pictures (detect_objects, point_objects, "
        "gaze_point need a moondream3 model with its region head) and analyze_video / classify_video "
        "for clips. The first call that needs weights loads them (seconds to a minute); pass "
        "model=<id> only when more than one model is available. Coordinates are normalised to 0..1."));
    reply_result(id, vqa::json::dump(result));
}

void handle_tools_call(const vqa::json::Value & id, const vqa::json::Value & params) {
    const std::string name = params.str("name");
    const vqa::json::Value * args_ptr = params.get("arguments");
    const vqa::json::Value empty = vqa::json::Value::make_object();
    const vqa::json::Value & args = args_ptr && args_ptr->is_object() ? *args_ptr : empty;

    for (const auto & t : kTools) {
        if (name != t.name) continue;
        std::string text, err;
        if (t.fn(args, text, err)) {
            reply_tool(id, text, false);
        } else {
            reply_tool(id, err.empty() ? std::string("tool failed") : err, true);
        }
        return;
    }
    reply_tool(id, "unknown tool '" + name + "'; see tools/list", true);
}

void handle_message(const std::string & line) {
    vqa::json::Value req;
    std::string perr;
    if (!vqa::json::parse(line, req, perr)) {
        reply_error(vqa::json::Value(), -32700, "parse error: " + perr);
        return;
    }
    if (!req.is_object()) {
        reply_error(vqa::json::Value(), -32600, "invalid request: expected an object");
        return;
    }
    const vqa::json::Value * id_ptr = req.get("id");
    const vqa::json::Value id = id_ptr ? *id_ptr : vqa::json::Value();
    const bool notification = (id_ptr == nullptr);   // no id -> no reply
    const std::string method = req.str("method");
    const vqa::json::Value empty = vqa::json::Value::make_object();
    const vqa::json::Value * p = req.get("params");
    const vqa::json::Value & params = p && p->is_object() ? *p : empty;

    if (method == "initialize") { handle_initialize(id, params); return; }
    if (method.rfind("notifications/", 0) == 0) return;   // initialized / cancelled / progress
    if (method == "ping") { reply_result(id, "{}"); return; }
    if (method == "tools/list") {
        reply_result(id, std::string("{\"tools\":") + tools_schema_compact() + "}");
        return;
    }
    if (method == "tools/call") {
        if (notification) return;
        handle_tools_call(id, params);
        return;
    }
    if (notification) return;
    reply_error(id, -32601, "method not found: " + method);
}

bool read_message(std::string & line) {
    line.clear();
    int c;
    while ((c = fgetc(stdin)) != EOF) {
        if (c == '\n') {
            if (line.empty()) continue;   // skip blank keep-alive lines
            // a UTF-8 BOM from a piped script would otherwise sit in front of
            // the first '{' and fail the parse
            if (line.size() >= 3 && (unsigned char)line[0] == 0xEF &&
                (unsigned char)line[1] == 0xBB && (unsigned char)line[2] == 0xBF)
                line.erase(0, 3);
            return !line.empty();
        }
        if (c != '\r') line += (char)c;
    }
    return !line.empty();
}

}  // namespace

namespace vqa {
int run_mcp_server(const std::string & models_dir, const std::string & tokenizer,
                   const std::string & preset_model) {
    if (!open_protocol_channel()) {
        fprintf(stderr, "mcp: cannot reserve stdout for JSON-RPC\n");
        return 1;
    }
    g_models_dir = models_dir.empty() ? std::string(".") : models_dir;
    g_preset_model = preset_model;
    mlog("SkyVision.cpp MCP server (stdio), models dir: %s", g_models_dir.c_str());
    g_reg.discover(g_models_dir, tokenizer);
    if (!preset_model.empty()) {
        std::string err;
        if (g_reg.find(preset_model)) {
            // already discovered - nothing to add
        } else if (!g_reg.add_path(preset_model, err)) {
            mlog("preset model '%s' ignored: %s", preset_model.c_str(), err.c_str());
            g_preset_model.clear();
        }
    }
    std::string line;
    while (read_message(line)) handle_message(line);
    mlog("stdin closed, exiting");
    g_reg.unload();
    return 0;
}
}  // namespace vqa
