// vqa_server.cpp - unified multi-model Web UI for vqa-cli --serve.
//
//   GET  /                      model list (only runnable backends are listed)
//   GET  /api/models            JSON model registry
//   GET  /model/<id>            per-model page (web/<arch>.html)
//   POST /api/model/<id>/load   load a model (one resident at a time)
//   POST /api/model/<id>/ask    body = media bytes (or empty); ?q=&mode=&fps=&max-frames=
//
// Models are discovered from --models-dir (default: cwd): every *.gguf whose
// general.architecture is a known text backend, with its runtime dependencies
// present (moondream3 wants a *-vision.gguf; both want tokenizer.json).
#include "vqa.hpp"
#include "http_server.hpp"
#include "media_io.hpp"

#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#endif

namespace {

struct ModelEntry {
    std::string id;          // file stem
    std::string path;        // gguf path
    std::string arch;        // moondream3 | videochat3
    std::string quant;       // from general.file_type / name heuristic
    uint64_t size = 0;
    std::string vision_path; // moondream only
    std::string tokenizer;     // resolved per arch
};

std::string g_models_dir = ".";
std::string g_web_dir = "web";
std::string g_tokenizer = "";        // optional override
std::vector<ModelEntry> g_models;
std::mutex g_mu;                      // serializes model access
std::string g_id_placeholder;         // substituted into pages per request
vqa::Model * g_loaded = nullptr;
std::string g_loaded_id;

bool file_exists(const std::string & p) {
#ifdef _WIN32
    return GetFileAttributesA(p.c_str()) != INVALID_FILE_ATTRIBUTES;
#else
    FILE * f = fopen(p.c_str(), "rb");
    if (f) { fclose(f); return true; }
    return false;
#endif
}

std::string read_file(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) return "";
    std::string out;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.resize(n);
    if (n) fread(out.data(), 1, n, f);
    fclose(f);
    return out;
}

std::string stem_of(const std::string & name) {
    size_t dot = name.rfind('.');
    return dot == std::string::npos ? name : name.substr(0, dot);
}

bool gguf_arch_of(const std::string & path, std::string & arch, std::string & ftype) {
    struct gguf_init_params p = { true, nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), p);
    if (!g) return false;
    int k = gguf_find_key(g, "general.architecture");
    if (k >= 0) arch = gguf_get_val_str(g, k);
    int kf = gguf_find_key(g, "general.file_type");
    if (kf >= 0) {
        char b[32];
        snprintf(b, sizeof(b), "f%d", (int)gguf_get_val_u32(g, kf));
        ftype = b;
    }
    gguf_free(g);
    return !arch.empty();
}

// pick a tokenizer.json for the arch: vc3 needs the Qwen one (vocab >= 150k);
// probe the models dir and ./models/* checkpoint dirs by vocab size
std::string resolve_tokenizer(const std::string & arch) {
    std::vector<std::string> cands = { g_models_dir + "/tokenizer.json" };
#ifdef _WIN32
    // models/<checkpoint>/tokenizer.json (fd.cFileName is a bare name, so
    // enumerate checkpoint dirs and probe each)
    WIN32_FIND_DATAA fdd;
    HANDLE hd = FindFirstFileA("models/*", &fdd);
    if (hd != INVALID_HANDLE_VALUE) {
        do {
            if (fdd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY && fdd.cFileName[0] != '.')
                cands.push_back(std::string("models/") + fdd.cFileName + "/tokenizer.json");
        } while (FindNextFileA(hd, &fdd));
        FindClose(hd);
    }
#endif
    const bool need_qwen = arch == "videochat3";
    for (auto & c : cands) {
        FILE * f = fopen(c.c_str(), "rb");
        if (!f) continue;
        std::string js;
        char buf[65536];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) js.append(buf, n);
        fclose(f);
        size_t vp = js.find("\"vocab\"");
        if (vp == std::string::npos) continue;
        size_t me = js.find("\"merges\"", vp);
        if (me == std::string::npos) me = js.size();
        size_t keys = std::count(js.begin() + vp, js.begin() + me, ':');
        if ((need_qwen && keys >= 150000) || (!need_qwen && keys >= 50000)) return c;
    }
    return "";
}

void discover_models() {
    g_models.clear();
    std::vector<std::string> files;
#ifdef _WIN32
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA((g_models_dir + "/*.gguf").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) files.push_back(fd.cFileName);
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    fprintf(stderr, "--serve model discovery is Windows-only for now\n");
    (void)files;
#endif
    // one shared vision/region file for all moondream weights
    std::string vision_path;
    for (auto & f : files)
        if (f.size() > 12 && f.substr(f.size() - 12) == "-vision.gguf") { vision_path = g_models_dir + "/" + f; break; }

    for (auto & f : files) {
        std::string arch, ftype;
        if (!gguf_arch_of(g_models_dir + "/" + f, arch, ftype)) continue;
        std::string tok = resolve_tokenizer(arch);
        if (!g_tokenizer.empty()) tok = g_tokenizer;
        if (arch == "moondream3") {
            if (vision_path.empty() || !file_exists(tok)) {
                fprintf(stderr, "serve: skip %s (needs %s + tokenizer.json)\n", f.c_str(),
                        vision_path.empty() ? "*-vision.gguf" : "tokenizer.json");
                continue;
            }
        } else if (arch == "videochat3") {
            if (!file_exists(tok)) {
                fprintf(stderr, "serve: skip %s (needs tokenizer.json)\n", f.c_str());
                continue;
            }
        } else {
            continue;   // vision/region side files and unknown archs stay unlisted
        }
        ModelEntry e;
        e.id = stem_of(f);
        e.path = g_models_dir + "/" + f;
        e.arch = arch;
        e.quant = ftype;
        e.vision_path = vision_path;
        e.tokenizer = tok;
        // size via filesystem
#ifdef _WIN32
        WIN32_FILE_ATTRIBUTE_DATA fa;
        if (GetFileAttributesExA(e.path.c_str(), GetFileExInfoStandard, &fa))
            e.size = ((uint64_t)fa.nFileSizeHigh << 32) | fa.nFileSizeLow;
#endif
        g_models.push_back(e);
    }
    fprintf(stderr, "serve: %zu runnable model(s) in %s\n", g_models.size(), g_models_dir.c_str());
}

const ModelEntry * find_model(const std::string & id) {
    for (auto & m : g_models) if (m.id == id) return &m;
    return nullptr;
}

bool ensure_loaded(const ModelEntry & e, std::string & err) {
    if (g_loaded && g_loaded_id == e.id) return true;
    if (g_loaded) {
        fprintf(stderr, "serve: unloading %s\n", g_loaded_id.c_str());
        delete g_loaded;    // frees the backend buffers (one model resident at a time)
        g_loaded = nullptr;
        g_loaded_id.clear();
    }
    fprintf(stderr, "serve: loading %s ...\n", e.id.c_str());
    fflush(stderr);
    vqa::Model * m = vqa::vqa_create_model(e.arch);
    if (!m) { err = "unknown backend"; return false; }
    vqa::LoadOptions opt;
    opt.vision_path = e.vision_path;
    if (!m->load(e.path, e.tokenizer.empty() ? "tokenizer.json" : e.tokenizer, opt)) {
        err = "model load failed";
        delete m;
        return false;
    }
    g_loaded = m;
    g_loaded_id = e.id;
    fprintf(stderr, "serve: %s ready\n", e.id.c_str());
    return true;
}

std::string models_json() {
    std::string s = "{\"models\":[";
    bool first = true;
    for (auto & m : g_models) {
        if (!first) s += ",";
        first = false;
        s += "{\"id\":\"" + vqa::http::json_escape(m.id) + "\",\"arch\":\"" + m.arch +
             "\",\"quant\":\"" + m.quant + "\",\"size\":" + std::to_string(m.size) +
             ",\"loaded\":" + (g_loaded_id == m.id ? "true" : "false") + "}";
    }
    return s + "]}";
}

bool handle_request(vqa::http::HttpSock c, const vqa::http::Request & req) {
    using namespace vqa::http;
    // --- API ---
    if (req.method == "GET" && req.path == "/api/models") { send_json(c, models_json()); return true; }

    if (req.path.rfind("/api/model/", 0) == 0) {
        std::string rest = req.path.substr(11);
        size_t slash = rest.find('/');
        std::string id = slash == std::string::npos ? rest : rest.substr(0, slash);
        std::string action = slash == std::string::npos ? "" : rest.substr(slash + 1);
        const ModelEntry * e = find_model(id);
        if (!e) { send_error(c, "unknown model: " + id); return true; }

        if (req.method == "POST" && action == "load") {
            std::lock_guard<std::mutex> lk(g_mu);
            std::string err;
            if (!ensure_loaded(*e, err)) { send(c, "500 Internal Server Error", "application/json", "{\"error\":\"" + json_escape(err) + "\"}"); return true; }
            send_json(c, "{\"ok\":true}");
            return true;
        }
        if (req.method == "POST" && action == "ask") {
            std::lock_guard<std::mutex> lk(g_mu);
            std::string err;
            if (!ensure_loaded(*e, err)) { send(c, "500 Internal Server Error", "application/json", "{\"error\":\"" + json_escape(err) + "\"}"); return true; }
            const bool stream = query_param(req.query, "stream") == "1";
            vqa::MediaInput media;
            const std::string kind = query_param(req.query, "kind");
            bool ok = false;
            if (!req.body.empty()) {
                if (kind == "video") {
                    // raw video file: dump to a temp file for the ffmpeg pipe
                    std::string tmp = "serve_upload.tmp";
                    FILE * f = fopen(tmp.c_str(), "wb");
                    if (f) { fwrite(req.body.data(), 1, req.body.size(), f); fclose(f); }
                    double fps = atof(query_param(req.query, "fps").c_str());
                    int mf = atoi(query_param(req.query, "max-frames").c_str());
                    ok = vqa::load_video_file(tmp, fps, mf, media);
                    remove(tmp.c_str());
                } else {
                    ok = vqa::load_image_from_memory(req.body, media);
                }
                if (!ok) { send_error(c, "media decode failed"); return true; }
            }
            vqa::GenParams gp;
            gp.max_tokens = atoi(query_param(req.query, "n").c_str());
            if (gp.max_tokens <= 0) gp.max_tokens = 256;
            std::string q = query_param(req.query, "q");
            std::string mode = query_param(req.query, "mode");
            if (q.empty() && media.empty()) { send_error(c, "need ?q= or a media body"); return true; }
            // stream state shared by the token callback and the final flush
            static thread_local std::string carry;
            static thread_local bool sent_any;
            carry.clear();
            sent_any = false;
            if (stream) {
                // NDJSON events: {"phase":"vision"} once the media is decoded,
                // {"t":".."} per generated piece, then {"done":true} / {"error":".."}
                // Chinese (any multibyte UTF-8) is often split across two tokens;
                // a raw partial sequence would render as U+FFFD forever, so only
                // complete characters are emitted and the tail is carried over.
                send_stream_head(c, "application/x-ndjson");
                send_chunk_str(c, "{\"phase\":\"vision\"}\n");
                carry.clear();
                sent_any = false;
                gp.on_token = [](const char * piece, void * ud) {
                    std::string s = carry + piece;
                    carry.clear();
                    size_t emit = s.size();
                    // find the last complete UTF-8 boundary: scan back at most
                    // 4 bytes (max sequence length) for an ASCII byte or a lead
                    // byte whose sequence has enough bytes after it
                    const size_t lo = s.size() > 4 ? s.size() - 4 : 0;
                    for (size_t k = s.size(); k > lo; k--) {
                        const unsigned char b = (unsigned char)s[k - 1];
                        if (b < 0x80) break;   // ASCII: everything before is complete
                        if (b >= 0xC0) {      // lead byte: complete only if not truncated
                            const size_t need = b >= 0xF0 ? 4 : b >= 0xE0 ? 3 : 2;
                            if (s.size() - (k - 1) < need) emit = k - 1;
                            break;
                        }
                        // continuation byte (0x80..0xBF): keep scanning back
                    }
                    if (emit < s.size()) { carry = s.substr(emit); s.resize(emit); }
                    if (!s.empty()) {
                        sent_any = true;
                        send_chunk_str(*(HttpSock *)ud, std::string("{\"t\":\"") + json_escape(s) + "\"}\n");
                    }
                };
                gp.on_token_ud = &c;
            }
            std::string out;
            if (!g_loaded->ask(media, q, mode, gp, out)) {
                if (stream) { send_chunk_str(c, "{\"error\":\"inference failed (see server log)\"}\n"); send_stream_end(c); }
                else send_error(c, "inference failed");
                return true;
            }
            if (stream) {
                // whole-result skills (detect/point/gaze) never emit token
                // events — deliver the text as one final event so the page
                // isn't left with an empty string ("未检测到目标")
                if (!sent_any && !out.empty())
                    send_chunk_str(c, std::string("{\"t\":\"") + json_escape(out) + "\"}\n");
                send_chunk_str(c, "{\"done\":true}\n");
                send_stream_end(c);
            }
            else send_json(c, "{\"text\":\"" + json_escape(out) + "\"}");
            return true;
        }
        send_error(c, "bad action");
        return true;
    }

    // --- pages ---
    auto serve_page = [&](const std::string & file) {
        std::string html = read_file(g_web_dir + "/" + file);
        if (html.empty()) { send(c, "404 Not Found", "text/plain", file + " not found under " + g_web_dir); return; }
        // inject the model id so page JS can address its API endpoints
        std::string marker = "__MODEL_ID__";
        size_t p;
        while ((p = html.find(marker)) != std::string::npos) html.replace(p, marker.size(), g_id_placeholder);
        send(c, "200 OK", "text/html; charset=utf-8", html);
    };

    if (req.method == "GET" && (req.path == "/" || req.path == "/index.html")) {
        std::string html = read_file(g_web_dir + "/home.html");
        if (html.empty()) { send(c, "404 Not Found", "text/plain", "web/home.html not found (run from the repo root)"); return true; }
        send(c, "200 OK", "text/html; charset=utf-8", html);
        return true;
    }
    if (req.method == "GET" && req.path.rfind("/model/", 0) == 0) {
        std::string id = req.path.substr(7);
        g_id_placeholder = id;
        const ModelEntry * e = find_model(id);
        if (!e) { send(c, "404 Not Found", "text/plain", "unknown model"); return true; }
        serve_page(e->arch == "moondream3" ? "moondream.html" : "vc3.html");
        return true;
    }
    send(c, "404 Not Found", "text/plain", "not found");
    return true;
}

std::string id_placeholder;   // set per page request before serve_page

}  // namespace

namespace vqa {
int run_web_ui(int port, const std::string & models_dir, const std::string & web_dir,
               const std::string & tokenizer, const char * bind_host) {
    g_models_dir = models_dir;
    g_web_dir = web_dir;
    g_tokenizer = tokenizer;
    discover_models();
    printf("SkyVision.cpp web UI: http://127.0.0.1:%d  (%zu models, Ctrl+C to stop)\n", port, g_models.size());
    fflush(stdout);
    return http::serve(port, handle_request, bind_host);
}
}  // namespace vqa
