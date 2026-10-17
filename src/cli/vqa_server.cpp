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
#include "model_registry.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#ifndef _WIN32
#   include <dirent.h>
#   include <sys/stat.h>
#endif
#include <map>
#include <set>
#include <deque>
#include <memory>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #ifndef _WIN32_WINNT
    #define _WIN32_WINNT 0x0601
  #endif
  #include <windows.h>
#endif

namespace {

using ModelEntry = vqa::ModelEntry;

std::string g_web_dir = "web";
std::mutex g_mu;                      // serializes model access
std::string g_id_placeholder;         // substituted into pages per request
std::string g_arch_placeholder;      // ditto, for arch-conditional UI

// discovery + the resident model slot, shared with the MCP front end
vqa::ModelRegistry g_reg;

// ---------------------------------------------------------------- realtime
// live stream session: one ffmpeg capture process + a resident inference loop
// that feeds new frames (and any queued user questions) into the model's
// incremental session every `interval` seconds. The model speaks on its own —
// the trained <|silence|> token means "nothing to say" and ends the turn.
// Clients observe via rt/events (poll with ?since=<seq> for NDJSON events).
#ifdef _WIN32
#ifndef STARTF_USESTDHANDLE
#define STARTF_USESTDHANDLE 0x100
#endif
struct CapProc {
    PROCESS_INFORMATION pi = {};
    HANDLE rh = INVALID_HANDLE_VALUE;
    bool spawn(const std::string & cmd) {
        SECURITY_ATTRIBUTES sa = { sizeof(sa), nullptr, TRUE };
        HANDLE wh;
        if (!CreatePipe(&rh, &wh, &sa, 0)) return false;
        SetHandleInformation(rh, HANDLE_FLAG_INHERIT, 0);
        // NOTE: no "cmd /c" here — cmd's quote-stripping rules mangle commands
        // with 3+ quotes (path + args); CreateProcessA parses them natively
        std::string full = cmd;
        STARTUPINFOA si = {};
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESTDHANDLE | STARTF_USESHOWWINDOW;
        si.hStdOutput = wh;
        // NUL for stdin/stderr: with STARTF_USESTDHANDLE all three handles
        // must be valid, and inheriting the console's breaks hidden starts
        HANDLE nul = CreateFileA("NUL", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_WRITE,
                                 &sa, OPEN_EXISTING, 0, nullptr);
        si.hStdError = nul;
        si.hStdInput = nul == INVALID_HANDLE_VALUE ? wh : nul;
        si.wShowWindow = SW_HIDE;
        std::vector<char> buf(full.begin(), full.end());
        buf.push_back(0);
        if (!CreateProcessA(nullptr, buf.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
            CloseHandle(rh); rh = INVALID_HANDLE_VALUE;
            return false;
        }
        CloseHandle(wh);   // child owns the write end now
        return true;
    }
    int read(char * out, int n) {
        DWORD got = 0;
        if (!ReadFile(rh, out, (DWORD)n, &got, nullptr)) return 0;
        return (int)got;
    }
    void stop() {
        if (pi.hProcess) { TerminateProcess(pi.hProcess, 0); WaitForSingleObject(pi.hProcess, 2000); }
        if (rh != INVALID_HANDLE_VALUE) { CloseHandle(rh); rh = INVALID_HANDLE_VALUE; }
        if (pi.hProcess) CloseHandle(pi.hProcess);
        if (pi.hThread) CloseHandle(pi.hThread);
        pi = {};
    }
};
#else
    #include <cstdio>
struct CapProc {
    FILE * p = nullptr;
    bool spawn(const std::string & cmd) { p = popen((cmd + " 2>/dev/null").c_str(), "r"); return p != nullptr; }
    int read(char * out, int n) { return p ? (int)fread(out, 1, n, p) : 0; }
    void stop() { if (p) pclose(p); p = nullptr; }
};
#endif

struct RtStream {
    std::string session, src;
    double interval = 3.0;
    int frames_per_turn = 1;
    bool use_sys_prompt = true;
    CapProc cap;
    std::thread reader, worker;
    std::mutex mu;
    std::deque<std::pair<std::vector<unsigned char>, double>> jpegs;  // pending (frame, t)
    std::deque<std::string> prompts;
    std::deque<std::string> events;   // NDJSON lines, oldest first
    long seq = 0;
    std::atomic<bool> stop_flag{false};
    bool started = false;

    void push_event(const std::string & line) {
        std::lock_guard<std::mutex> lk(mu);
        // embed the sequence number as the first field so rt/events can page
        events.push_back("{\"seq\":" + std::to_string(++seq) + "," + line.substr(1));
        if (events.size() > 4096) events.erase(events.begin(), events.end() - 4096);
    }
    ~RtStream() {
        stop_flag = true;
        cap.stop();
        if (reader.joinable()) reader.join();
        if (worker.joinable()) worker.join();
    }
};
static std::map<std::string, std::shared_ptr<RtStream>> g_rt;
static std::mutex g_rt_mu;

static std::string ffmpeg_path() {
    if (const char * e = getenv("FFMPEG")) return e;
    return "ffmpeg";
}

// the official real_time_generate system prompt: makes the model narrate what
// the camera sees and stay silent when nothing relevant happens
static const char * RT_SYSTEM_PROMPT =
    "You are a helpful AI assistant. You perceive and understand the surrounding environment in real-time "
    "through the camera and interact with the user. Whether or not the user actively asks questions, you "
    "continuously observe and analyze visual information, maintaining awareness of the environment.\n\n"
    "- Continuous Perception: Always observe and analyze the visual information captured by the camera.\n"
    "- Observation-Based Answers: All answers must be based on visual observations, including historical "
    "and real-time data.\n"
    "- If you notice significant changes, promptly update your answers; when you cannot answer or have "
    "finished, output <|silence|>.";

static void rt_reader_loop(std::shared_ptr<RtStream> s) {
    std::vector<unsigned char> buf;
    char tmp[1 << 16];
    const auto t0 = std::chrono::steady_clock::now();
    size_t frame_start = std::string::npos;
    while (!s->stop_flag) {
        int got = s->cap.read(tmp, sizeof(tmp));
        if (got <= 0) {   // source ended or died
            s->push_event("{\"event\":\"source_end\"}\n");
            break;
        }
        buf.insert(buf.end(), tmp, tmp + got);
        // split the mjpeg stream on SOI(FFD8)/EOI(FFD9) markers
        for (size_t i = 1; i < buf.size(); i++) {
            if (buf[i - 1] == 0xFF && buf[i] == 0xD9 && frame_start != std::string::npos) {
                if (i + 1 - frame_start > 64) {
                    double t = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0).count();
                    std::vector<unsigned char> jpg(buf.begin() + frame_start, buf.begin() + i + 1);
                    {
                        std::lock_guard<std::mutex> lk(s->mu);
                        s->jpegs.push_back({jpg, t});
                        while (s->jpegs.size() > 16) s->jpegs.pop_front();   // keep realtime
                    }
                }
                buf.erase(buf.begin(), buf.begin() + i + 1);
                frame_start = std::string::npos;
                i = 0;
                continue;
            }
            if (buf[i - 1] == 0xFF && buf[i] == 0xD8 && frame_start == std::string::npos)
                frame_start = i - 1;
        }
    }
}

static void rt_worker_loop(std::shared_ptr<RtStream> s, std::string model_id) {
    bool first = true;
    while (!s->stop_flag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if (s->stop_flag) break;
        // wait out the turn interval
        // (checked in 200ms slices so stop/say react quickly)
        static thread_local double since = 0.0;
        since += 0.2;
        if (since < s->interval) continue;
        since = 0.0;
        // drain the pending frames + prompts. A queued question goes out on
        // its own turn (frames wait for the next one): the realtime SFT is
        // touchy about prompt+new-frame turns and reliably answers text-only
        // questions while narration stays on frame-only turns.
        vqa::MediaInput media;
        media.kind = vqa::MediaKind::Video;
        std::string q;
        {
            std::lock_guard<std::mutex> lk(s->mu);
            bool has_q = !s->prompts.empty();
            while (!s->prompts.empty()) {
                if (!q.empty()) q += "\n";
                q += s->prompts.front();
                s->prompts.pop_front();
            }
            if (has_q) { /* text-only question turn */ }
            else {
            int n = std::min<int>((int)s->jpegs.size(), std::max(1, s->frames_per_turn));
            for (int i = 0; i < n; i++) {
                vqa::MediaInput one;
                if (vqa::load_image_from_memory(
                        std::string(s->jpegs.front().first.begin(), s->jpegs.front().first.end()), one)) {
                    if (one.frames.empty()) continue;
                    media.frames.push_back(std::move(one.frames[0]));
                    media.width = one.width;
                    media.height = one.height;
                    media.timestamps.push_back(s->jpegs.front().second);
                }
                s->jpegs.pop_front();
            }
            }
        }
        if (media.frames.empty() && q.empty()) continue;   // nothing new to say
        // model speaking on its own: frames-only turns, silence ends them
        vqa::GenParams gp;
        gp.max_tokens = 200;
        std::string carry;
        gp.on_token = [](const char * piece, void * ud) {
            ((RtStream *)ud)->push_event(std::string("{\"t\":\"") +
                                         vqa::http::json_escape(piece) + "\"}\n");
        };
        gp.on_token_ud = s.get();
        if (first) {
            first = false;
            // the realtime SFT goes mute (and stays mute for later questions)
            // when turn 1 carries no user text — seed one so the narration
            // lands as a normal answer and the session stays responsive
            if (q.empty()) q = "Please observe and describe what you see.";
            if (s->use_sys_prompt)
                gp.history = std::string("<|im_start|>system\n") + RT_SYSTEM_PROMPT + "<|im_end|>\n";
        }
        std::string out;
        s->push_event(q.empty() ? "{\"event\":\"observe\"}\n"
                                : std::string("{\"event\":\"ask\",\"q\":\"") + vqa::http::json_escape(q) + "\"}\n");
        {
            std::lock_guard<std::mutex> lk(g_mu);
            vqa::Model * m = g_reg.loaded();
            if (!m || g_reg.loaded_id() != model_id) {
                s->push_event("{\"error\":\"model no longer loaded\"}\n");
                break;
            }
            if (!m->ask_session(media, q, gp, out)) {
                s->push_event("{\"error\":\"turn failed\"}\n");
                continue;
            }
        }
        s->push_event(out.empty() ? "{\"event\":\"silence\"}\n"
                                  : "{\"event\":\"turn_end\"}\n");
    }
}

static bool rt_start(const std::string & session, const std::string & src,
                     double interval, int frames_per_turn, bool use_sys_prompt,
                     std::string & err) {
    std::lock_guard<std::mutex> lk(g_rt_mu);
    if (g_rt.count(session)) { err = "session already streaming"; return false; }
    auto s = std::make_shared<RtStream>();
    s->session = session;
    s->src = src;
    s->interval = interval > 0.2 ? interval : 3.0;
    s->frames_per_turn = frames_per_turn > 0 ? frames_per_turn : 1;
    s->use_sys_prompt = use_sys_prompt;
    // src: "video=..." -> Windows dshow camera; anything else is a path/URL
    std::string input;
    if (src.rfind("video=", 0) == 0)
        input = "-f dshow -i \"" + src + "\"";
    else
        input = "-re -i \"" + src + "\"";
    const double fps = 1.0 / s->interval;
    char tail[160];
    snprintf(tail, sizeof(tail), "-vf fps=%.3f -f image2pipe -vcodec mjpeg -", fps);
    std::string cmd = "\"" + ffmpeg_path() + "\" -nostdin -hide_banner -loglevel error " + input + " " + tail;
    if (!s->cap.spawn(cmd)) { err = "cannot spawn ffmpeg (check FFMPEG / source)"; return false; }
    s->started = true;
    g_rt[session] = s;
    s->reader = std::thread(rt_reader_loop, s);
    s->worker = std::thread(rt_worker_loop, s, g_reg.loaded_id());
    return true;
}

static void rt_stop(const std::string & session) {
    std::shared_ptr<RtStream> s;
    {
        std::lock_guard<std::mutex> lk(g_rt_mu);
        auto it = g_rt.find(session);
        if (it == g_rt.end()) return;
        s = it->second;
        g_rt.erase(it);
    }
    s->stop_flag = true;
    s->cap.stop();
    if (s->reader.joinable()) s->reader.join();
    if (s->worker.joinable()) s->worker.join();
    if (g_reg.loaded()) g_reg.loaded()->session_reset();
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

// ---------------------------------------------------------------- models
// Discovery and the single-resident model slot live in core/model_registry.cpp,
// shared with the MCP front end: one place decides which GGUFs are runnable and
// which side files (vision tower / tokenizer.json) each one needs.
const ModelEntry * find_model(const std::string & id) { return g_reg.find(id); }
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
    // legacy qwen3.5/marlin ggufs carry swapped arch labels; marlin.skills
    // (written by the converter from the checkpoint's architectures) is
    // authoritative for which fine-tune it is
    int ks = gguf_find_key(g, "marlin.skills");
    if (arch == "qwen3_5" || arch == "marlin") {
        // skills=true -> Marlin fine-tune; false or key missing (older
        // converter that labelled plain qwen3.5 files "marlin") -> qwen3_5
        arch = (ks >= 0 && gguf_get_val_bool(g, ks)) ? "marlin" : "qwen3_5";
    }
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
#else
    // models/<checkpoint>/tokenizer.json: enumerate checkpoint dirs and probe each
    DIR * dd = opendir("models");
    if (dd) {
        struct dirent * de;
        while ((de = readdir(dd))) {
            if (de->d_name[0] == '.') continue;
            std::string c = std::string("models/") + de->d_name + "/tokenizer.json";
            struct stat st;
            if (stat(c.c_str(), &st) == 0 && S_ISREG(st.st_mode)) cands.push_back(c);
        }
        closedir(dd);
    }
#endif
    const bool need_qwen = arch == "videochat3" || arch == "qwen3_5" || arch == "marlin" || arch == "moss_vl" || arch == "mage_vl";
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
    DIR * d = opendir(g_models_dir.c_str());
    if (!d) { fprintf(stderr, "serve: models dir not found: %s\n", g_models_dir.c_str()); return; }
    struct dirent * e;
    while ((e = readdir(d)))
        if (e->d_name[0] != '.') {
            std::string name = e->d_name;
            if (name.size() > 5 && name.substr(name.size() - 5) == ".gguf") files.push_back(name);
        }
    closedir(d);
    std::sort(files.begin(), files.end());
#endif
    // one shared vision/region file for all moondream weights
    std::string vision_path;
    for (auto & f : files)
        if (f.size() > 12 && f.substr(f.size() - 12) == "-vision.gguf") { vision_path = g_models_dir + "/" + f; break; }

    for (auto & f : files) {
        std::string arch, ftype;
        if (!gguf_arch_of(g_models_dir + "/" + f, arch, ftype)) continue;
        // per-model tokenizer wins: models/<gguf-stem>/tokenizer.json (the
        // generic resolver can pick a same-vocab sibling whose added tokens
        // differ - moss_vl's <|image_pad|>/<|silence|> only exist in its own)
        std::string own = std::string("models/") + stem_of(f) + "/tokenizer.json";
        std::string tok = file_exists(own) ? own : resolve_tokenizer(arch);
        if (!g_tokenizer.empty()) tok = g_tokenizer;
        if (arch == "moondream3") {
            if (vision_path.empty() || !file_exists(tok)) {
                fprintf(stderr, "serve: skip %s (needs %s + tokenizer.json)\n", f.c_str(),
                        vision_path.empty() ? "*-vision.gguf" : "tokenizer.json");
                continue;
            }
        } else if (arch == "videochat3" || arch == "qwen3_5" || arch == "marlin" || arch == "moss_vl" || arch == "mage_vl") {
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
#else
        struct stat st;
        if (stat(e.path.c_str(), &st) == 0) e.size = (uint64_t)st.st_size;
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
    if (g_reg.loaded() && g_reg.loaded_id() == e.id) return true;
    return g_reg.load(e, vqa::LoadOptions(), err) != nullptr;
}

// the page JS reads id/arch/quant/size/loaded; the MCP front end builds a richer
// shape from the same entries (ModelRegistry::models_json)
std::string models_json() {
    std::string s = "{\"models\":[";
    bool first = true;
    for (const auto & m : g_reg.entries()) {
        if (!first) s += ",";
        first = false;
        s += "{\"id\":\"" + vqa::http::json_escape(m.id) + "\",\"arch\":\"" + m.arch +
             "\",\"quant\":\"" + m.quant + "\",\"size\":" + std::to_string(m.size) +
             ",\"loaded\":" + (g_reg.loaded_id() == m.id ? "true" : "false") + "}";
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

        if (action.rfind("rt/", 0) == 0) {
            const std::string sess = query_param(req.query, "session");
            if (sess.empty()) { send_error(c, "need ?session="); return true; }
            if (action == "rt/start" && req.method == "POST") {
                std::lock_guard<std::mutex> lk(g_mu);
                std::string err;
                if (!ensure_loaded(*e, err)) { send_error(c, err); return true; }
                g_reg.loaded()->session_reset();   // a new stream always starts fresh
                std::string src = query_param(req.query, "src");
                if (src.empty() && !req.body.empty()) {
                    // uploaded video = the stream itself (plays in realtime with -re)
                    static std::string up = "rt_upload.tmp";
                    FILE * f = fopen(up.c_str(), "wb");
                    if (!f) { send_error(c, "cannot write upload"); return true; }
                    fwrite(req.body.data(), 1, req.body.size(), f);
                    fclose(f);
                    src = up;
                }
                if (src.empty()) { send_error(c, "need ?src=<camera|path|url> or a video body"); return true; }
                double interval = atof(query_param(req.query, "interval").c_str());
                int fpt = atoi(query_param(req.query, "frames").c_str());
                std::string rerr;
                // the official realtime system prompt is an admitted placeholder
                // (train/inference divergence suppresses speech) - off by default
                const bool sysp = query_param(req.query, "sys") == "1";
                if (!rt_start(sess, src, interval, fpt, sysp, rerr)) { send_error(c, rerr); return true; }
                send_json(c, "{\"ok\":true}");
                return true;
            }
            if (action == "rt/stop" && req.method == "POST") {
                rt_stop(sess);
                send_json(c, "{\"ok\":true}");
                return true;
            }
            if (action == "rt/say" && req.method == "POST") {
                std::lock_guard<std::mutex> lk(g_rt_mu);
                auto it = g_rt.find(sess);
                if (it == g_rt.end()) { send_error(c, "no such streaming session"); return true; }
                std::string q = query_param(req.query, "q");
                if (q.empty()) { send_error(c, "need ?q="); return true; }
                {
                    std::lock_guard<std::mutex> lk2(it->second->mu);
                    it->second->prompts.push_back(q);
                }
                send_json(c, "{\"ok\":true}");
                return true;
            }
            if (action == "rt/events" && req.method == "GET") {
                long since = atol(query_param(req.query, "since").c_str());
                std::shared_ptr<RtStream> s;
                {
                    std::lock_guard<std::mutex> lk(g_rt_mu);
                    auto it = g_rt.find(sess);
                    if (it != g_rt.end()) s = it->second;
                }
                std::string body;
                long last = since;
                if (s) {
                    std::lock_guard<std::mutex> lk(s->mu);
                    for (const auto & ev : s->events) {
                        // events carry their seq as the first field: {"seq":N,...
                        long eseq = atol(ev.c_str() + 7);
                        if (eseq > since) { body += ev; last = eseq; }
                    }
                }
                if (body.empty()) { send_json(c, "{\"events\":[],\"since\":" + std::to_string(last) + "}"); return true; }
                send(c, "200 OK", "application/x-ndjson", body);
                return true;
            }
            send_error(c, "bad rt action");
            return true;
        }

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
            const std::string kind = query_param(req.query, "kind");
            // raw-upload cache: decode (image parse or an ffmpeg pipe for videos)
            // ran on EVERY request even when the vision features were cached -
            // and video decode costs seconds. Key the decoded frames on the
            // uploaded bytes + decode parameters instead.
            static struct {
                uint64_t key = 0;
                vqa::MediaInput media;
            } upload_cache;
            uint64_t ukey = 1469598103934665603ull;
            for (unsigned char ch : req.body) { ukey ^= ch; ukey *= 1099511628211ull; }
            ukey ^= (uint64_t)req.body.size() ^ (kind == "video" ? 0x9e3779b97f4a7c15ull : 0);
            const bool want_video = (kind == "video");
            double fps_d = 0; int mf_i = 0;
            if (want_video) {
                fps_d = atof(query_param(req.query, "fps").c_str());
                mf_i = atoi(query_param(req.query, "max-frames").c_str());
                ukey ^= (uint64_t)(int)(fps_d * 1000) << 20 ^ (uint64_t)mf_i;
            }
            vqa::MediaInput media;
            bool ok = false;
            if (!req.body.empty() && upload_cache.key == ukey && !upload_cache.media.empty()) {
                media = upload_cache.media;
                ok = true;
            } else if (!req.body.empty()) {
                if (want_video) {
                    std::string tmp = "serve_upload.tmp";
                    FILE * f = fopen(tmp.c_str(), "wb");
                    if (f) { fwrite(req.body.data(), 1, req.body.size(), f); fclose(f); }
                    ok = vqa::load_video_file(tmp, fps_d, mf_i, media);
                    remove(tmp.c_str());
                } else {
                    ok = vqa::load_image_from_memory(req.body, media);
                }
                if (ok) { upload_cache.key = ukey; upload_cache.media = media; }
            }
            if (!req.body.empty() && !ok) {
                if (stream) {
                    send_stream_head(c, "application/x-ndjson");
                    send_chunk_str(c, "{\"error\":\"media decode failed (video needs ffmpeg; check FFMPEG)\"}\n");
                    send_stream_end(c);
                } else send_error(c, "media decode failed");
                return true;
            }
            vqa::GenParams gp;
            gp.max_tokens = atoi(query_param(req.query, "n").c_str());
            if (gp.max_tokens <= 0) gp.max_tokens = 256;
            std::string q = query_param(req.query, "q");
            std::string mode = query_param(req.query, "mode");
            if (q.empty() && media.empty()) { send_error(c, "need ?q= or a media body"); return true; }
            // --- multi-turn sessions (?session=<id>, ?reset=1 clears) ---
            static std::map<std::string, std::vector<std::pair<std::string, std::string>>> sessions;
            const std::string sess = query_param(req.query, "session");
            const bool realtime = query_param(req.query, "rt") == "1";   // incremental KV session
            if (!sess.empty() && query_param(req.query, "reset") == "1") {
                sessions.erase(sess);
                if (realtime) g_reg.loaded()->session_reset();
            }
            if (sess.empty() && query_param(req.query, "reset") == "1" && q.empty()) {
                send_json(c, "{\"ok\":true}"); return true;
            }
            if (realtime) {
                // the engine keeps ONE live session; a fresh HTTP session name
                // must reset it, otherwise turn 1 splices onto a stranger's KV
                static std::set<std::string> rt_sessions;
                if (!sess.empty() && !rt_sessions.count(sess)) {
                    rt_sessions.insert(sess);
                    g_reg.loaded()->session_reset();
                }
                if (!sess.empty() && query_param(req.query, "reset") == "1")
                    rt_sessions.erase(sess);
            }
            // rebuild the conversation prefix from the stored turns (text-only:
            // media from earlier turns is not re-sent; ask about it again)
            if (!sess.empty() && !realtime) {   // rt sessions carry context engine-side
                auto & turns = sessions[sess];
                while (turns.size() > 12) turns.erase(turns.begin());   // keep the last 12 turns
                std::string hist;
                for (const auto & t : turns) {
                    std::string a = t.second;
                    if (a.size() > 1500) a.resize(1500);
                    hist += "<|im_start|>user\n" + t.first + "<|im_end|>\n<|im_start|>assistant\n"
                         + a + "<|im_end|>\n";
                }
                gp.history = hist;
            }
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
                if (stream) {
                    gp.on_stage = [](const char * stage, int done, int total, void * ud) {
                        char ev[96];
                        int n = snprintf(ev, sizeof(ev), "{\"stage\":\"%s\",\"done\":%d,\"total\":%d}", stage, done, total);
                        if (n > 0) { ev[n] = 10; ev[n+1] = 0; send_chunk_str(*(HttpSock *)ud, ev); }
                    };
                    gp.on_stage_ud = &c;
                }
            }
            std::string out;
            vqa::Model * m = g_reg.loaded();
            const bool ran = !m ? false
                : realtime && sess.empty()
                    ? false
                    : (realtime ? m->ask_session(media, q, gp, out)
                                : m->ask(media, q, mode, gp, out));
            if (!ran) {
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
            if (!sess.empty() && !q.empty())
                sessions[sess].push_back({q, out});   // remember the turn
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
        marker = "__MODEL_ARCH__";
        while ((p = html.find(marker)) != std::string::npos) html.replace(p, marker.size(), g_arch_placeholder);
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
        g_arch_placeholder = e->arch;
        serve_page(e->arch == "moondream3" ? "moondream.html"
                 : e->arch == "moss_vl" ? "moss.html"
                 : "vc3.html");   // videochat3 / qwen3_5 / marlin / videomae
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
    g_web_dir = web_dir;
    g_reg.discover(models_dir, tokenizer);   // same discovery as the MCP server
    printf("SkyVision.cpp web UI: http://127.0.0.1:%d  (%zu models, Ctrl+C to stop)\n", port,
           g_reg.entries().size());
    fflush(stdout);
    return http::serve(port, handle_request, bind_host);
}
}  // namespace vqa
