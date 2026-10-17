// moondream_server.inc — minimal embedded web UI for the moondream CLI.
// Included from src/models/moondream/moondream.cpp after Engine/VisionModel are available.
// Philosophy of the project: no external dependencies — hand-rolled HTTP on
// plain sockets, sequential request handling (single local user).
//
//   GET  /                    → single-page UI (embedded HTML)
//   POST /api/ask             → body {"prompt": "...", "n": 128} → {"text": "..."}
//   POST /api/vision?q=...&mode=ask|detect&obj=...
//        body = raw image bytes (PNG/JPEG, decoded by stb_image) → {"text": "..."}
//        or {"boxes": [[x0,y0,x1,y1], ...]} (normalized, for mode=detect)

#ifdef _WIN32
#   include <winsock2.h>
#   include <ws2tcpip.h>
#   pragma comment(lib, "ws2_32.lib")
   typedef int socklen_t;
#else
#   include <sys/socket.h>
#   include <netinet/in.h>
#   include <unistd.h>
#   include <arpa/inet.h>
#   define SOCKET int
#   define INVALID_SOCKET (-1)
#   define closesocket close
#endif

#include <map>
#include <string_view>
#include <array>
#include <sstream>

static void md_url_decode(std::string & s) {
    std::string o; o.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '%' && i + 2 < s.size()) {
            char h[3] = { s[i+1], s[i+2], 0 };
            o += (char)strtol(h, nullptr, 16);
            i += 2;
        } else if (s[i] == '+') o += ' ';
        else o += s[i];
    }
    s = o;
}

static std::string md_json_escape(const std::string & s) {
    std::string o; o.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': break;
            case '\t': o += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
                else o += c;
        }
    }
    return o;
}

// tiny JSON "string field" extractor: finds "key":"value" (no escapes in value assumed)
static std::string md_json_field(const std::string & body, const char * key) {
    std::string pat = std::string("\"") + key + "\"";
    size_t p = body.find(pat);
    if (p == std::string::npos) return "";
    p = body.find(':', p + pat.size());
    if (p == std::string::npos) return "";
    p = body.find('"', p);
    if (p == std::string::npos) return "";
    size_t e = body.find('"', p + 1);
    if (e == std::string::npos) return "";
    return body.substr(p + 1, e - p - 1);
}

static long md_json_int(const std::string & body, const char * key, long dflt) {
    std::string pat = std::string("\"") + key + "\"";
    size_t p = body.find(pat);
    if (p == std::string::npos) return dflt;
    p = body.find(':', p + pat.size());
    if (p == std::string::npos) return dflt;
    return strtol(body.c_str() + p + 1, nullptr, 10);
}

// ---- generation helpers (shared by /api/ask and /api/vision) ----

static std::string md_generate_text(Engine & e, const std::string & prompt, int n_gen) {
    std::vector<int> toks = { 0, 1, 15381, 2 };
    auto q = e.tok.encode(prompt);
    toks.insert(toks.end(), q.begin(), q.end());
    toks.push_back(3);
    std::vector<int> out;
    int cur = e.step(toks, nullptr);
    out.push_back(cur);
    for (int i = 1; i < n_gen; i++) {
        cur = e.step({ cur }, nullptr);
        if (cur == 0) break;
        out.push_back(cur);
    }
    return e.tok.decode(out);
}

// image → [bos;729] embedding prefix → prefill (bidirectional) → prompt prefill
static void md_prefill_image(Engine & e, VisionModel & vm, const uint8_t * rgb, int w, int h,
                             std::vector<float> & prefix_emb) {
    int th = 0, tw = 0, n_crops = 0;
    std::vector<float> crops = mdprep::prepare_crops(rgb, w, h, &th, &tw, &n_crops);
    std::vector<float> enc = vm.encode_crops(crops.data(), n_crops, vm.n_layers, e.backend);
    std::vector<float> img_emb = vm.project(enc.data(), n_crops, th, tw, e.backend);
    const int D = e.model.n_embd;
    std::vector<ggml_fp16_t> bos16(D);
    ggml_backend_tensor_get(e.model.tok_embd, bos16.data(), 0, sizeof(ggml_fp16_t) * D);
    prefix_emb.resize((size_t)(1 + 729) * D);
    for (int d = 0; d < D; d++) prefix_emb[d] = ggml_fp16_to_fp32(bos16[d]);
    memcpy(prefix_emb.data() + D, img_emb.data(), sizeof(float) * (size_t)729 * D);
    e.step({}, nullptr, prefix_emb.data(), 730, nullptr, true);
}

// ---- cross-request caches ---------------------------------------------------
// same-image cache: repeating queries on one image (the typical web UI flow)
// skip both the SigLIP encode AND the 730-token prefill by rewinding n_past
// to 730 — the KV slots beyond it are simply overwritten by the next prompt.
// same-question cache: repeated BPE encode of identical question text.
struct MdReqCache {
    uint64_t img_key = 0;             // FNV-1a of raw image body
    std::vector<float> prefix;        // [730, D] bos+img embeddings
    bool kv_valid = false;            // engine KV currently holds this prefix
    uint64_t q_key = 0;
    std::vector<int> q_toks;
};
static MdReqCache md_cache;

static uint64_t md_fnv(std::string_view s) {
    uint64_t h = 1469598103934665603ull;
    for (char c : s) { h ^= (unsigned char)c; h *= 1099511628211ull; }
    return h;
}

// make sure the engine holds THIS image's prefix at KV [0,730); n_past ends at 730
static void md_ensure_img_prefill(Engine & e, VisionModel & vm, const uint8_t * rgb, int w, int h) {
    // the key must actually gate the reuse: the web UI sends a different image
    // per request, and reusing the first image's prefix made every answer
    // describe the old picture
    uint64_t key = md_fnv(std::string_view((const char *)rgb, (size_t)w * h * 3));
    key ^= (uint64_t)w << 32 ^ (uint64_t)h;
    if (md_cache.img_key == key && md_cache.kv_valid) { e.n_past = 730; return; }   // rewind: reuse KV prefix
    if (md_cache.img_key == key && !md_cache.prefix.empty()) {
        e.step({}, nullptr, md_cache.prefix.data(), 730, nullptr, true);   // engine was reset, prefix hot
        md_cache.kv_valid = true;
        return;
    }
    e.reset();   // different image: clear the old prefix before re-encoding
    md_cache.prefix.clear();
    md_prefill_image(e, vm, rgb, w, h, md_cache.prefix);   // encode + prefill + fill cache
    md_cache.img_key = key;
    md_cache.q_key = 0;   // question cache is only valid within one image
    md_cache.kv_valid = true;
}

static std::string md_vqa(Engine & e, VisionModel & vm, const uint8_t * rgb, int w, int h,
                          const std::string & question, int n_gen) {
    md_ensure_img_prefill(e, vm, rgb, w, h);
    std::vector<int> toks = { 1, 15381, 2 };
    uint64_t qk = md_fnv(question);
    if (qk == md_cache.q_key && !md_cache.q_toks.empty()) {
        toks.insert(toks.end(), md_cache.q_toks.begin(), md_cache.q_toks.end());
    } else {
        md_cache.q_toks = e.tok.encode(question);
        md_cache.q_key = qk;
        toks.insert(toks.end(), md_cache.q_toks.begin(), md_cache.q_toks.end());
    }
    toks.push_back(3);
    std::vector<int> out;
    int cur = e.step(toks, nullptr);
    out.push_back(cur);
    for (int i = 1; i < n_gen; i++) {
        cur = e.step({ cur }, nullptr);
        if (cur == 0) break;
        out.push_back(cur);
    }
    return e.tok.decode(out);
}

static std::vector<std::array<float,4>> md_detect(Engine & e, VisionModel & vm, RegionModel & rm,
                                                  const uint8_t * rgb, int w, int h,
                                                  const std::string & obj) {
    md_ensure_img_prefill(e, vm, rgb, w, h);
    std::vector<int> dtoks = { 1, 7235, 476, 2 };
    std::string o = " " + obj;
    auto oq = e.tok.encode(o);
    dtoks.insert(dtoks.end(), oq.begin(), oq.end());
    dtoks.push_back(3);
    const int D = e.model.n_embd;
    std::vector<float> hidden(D), emb(D), lv(e.model.n_vocab);
    int next = e.step(dtoks, lv.data(), nullptr, 0, hidden.data(), false);
    next = (lv[0] >= lv[5]) ? 0 : 5;
    std::vector<std::array<float,4>> boxes;
    while (next != 0 && boxes.size() < 20) {
        float xl[1024];
        rm.decode_coordinate(hidden.data(), xl);
        float xc = RegionModel::logits_to_center(xl, 1024);
        rm.encode_coordinate(&xc, 1, emb.data());
        e.step({}, nullptr, emb.data(), 1, hidden.data(), false);
        rm.decode_coordinate(hidden.data(), xl);
        float yc = RegionModel::logits_to_center(xl, 1024);
        rm.encode_coordinate(&yc, 1, emb.data());
        e.step({}, nullptr, emb.data(), 1, hidden.data(), false);
        float sl[2048];
        rm.decode_size(hidden.data(), sl);
        int bins[2] = { vqa::greedy_argmax(sl, 1024), vqa::greedy_argmax(sl + 1024, 1024) };
        float sizes[2];
        RegionModel::bins_to_size(bins, sizes);
        rm.encode_size(sizes, 1, emb.data());
        // same single-step fix as md_point: logits and hidden come from one call
        e.step({}, lv.data(), emb.data(), 1, hidden.data(), false);
        boxes.push_back({{ xc - sizes[0] / 2, yc - sizes[1] / 2, xc + sizes[0] / 2, yc + sizes[1] / 2 }});
        next = (lv[0] >= lv[5]) ? 0 : 5;
    }
    return boxes;
}

// rows of the token embedding table -> raw f32 embeddings (f16 table)
static void md_tok_embs(Engine & e, const std::vector<int> & toks, std::vector<float> & out) {
    const int D = e.model.n_embd;
    out.resize(toks.size() * D);
    std::vector<ggml_fp16_t> row(D);
    for (size_t i = 0; i < toks.size(); i++) {
        ggml_backend_tensor_get(e.model.tok_embd, row.data(),
                                (size_t)toks[i] * D * sizeof(ggml_fp16_t), sizeof(ggml_fp16_t) * D);
        for (int d = 0; d < D; d++) out[i * D + d] = ggml_fp16_to_fp32(row[d]);
    }
}

// official point skill: template [1,2581,2] + " <obj>" + [3], no size, N points
static std::vector<std::array<float,2>> md_point(Engine & e, VisionModel & vm, RegionModel & rm,
                                                 const uint8_t * rgb, int w, int h,
                                                 const std::string & obj) {
    md_ensure_img_prefill(e, vm, rgb, w, h);
    std::vector<int> dtoks = { 1, 2581, 2 };
    std::string o = " " + obj;
    auto oq = e.tok.encode(o);
    dtoks.insert(dtoks.end(), oq.begin(), oq.end());
    dtoks.push_back(3);
    const int D = e.model.n_embd;
    std::vector<float> hidden(D), emb(D), lv(e.model.n_vocab);
    int next = e.step(dtoks, lv.data(), nullptr, 0, hidden.data(), false);
    next = (lv[0] >= lv[5]) ? 0 : 5;
    std::vector<std::array<float,2>> pts;
    while (next != 0 && pts.size() < 20) {
        float xl[1024];
        rm.decode_coordinate(hidden.data(), xl);
        float xc = RegionModel::logits_to_center(xl, 1024);
        rm.encode_coordinate(&xc, 1, emb.data());
        e.step({}, nullptr, emb.data(), 1, hidden.data(), false);
        rm.decode_coordinate(hidden.data(), xl);
        float yc = RegionModel::logits_to_center(xl, 1024);
        rm.encode_coordinate(&yc, 1, emb.data());
        // ONE step yields both the next hidden and the eos/coord logits - the
        // old code fed the y-embedding twice (once for hidden, once for logits),
        // shifting every following position and producing marching garbage points
        e.step({}, lv.data(), emb.data(), 1, hidden.data(), false);
        pts.push_back({{ xc, yc }});
        next = (lv[0] >= lv[5]) ? 0 : 5;
    }
    return pts;
}

// official detect_gaze skill: embedding-level prompt
//   emb("\n\nPoint:") + enc(x) + enc(y) + emb(" gaze\n\n")
// then eos -> none, else one predicted (x,y) fixation point
static bool md_gaze(Engine & e, VisionModel & vm, RegionModel & rm,
                    const uint8_t * rgb, int w, int h, float sx, float sy,
                    float & gx, float & gy) {
    md_ensure_img_prefill(e, vm, rgb, w, h);
    const int D = e.model.n_embd;
    std::vector<float> a, b;
    md_tok_embs(e, e.tok.encode("\n\nPoint:"), a);
    md_tok_embs(e, e.tok.encode(" gaze\n\n"), b);
    std::vector<float> xemb(D), yemb(D);   // encode_coordinate projects into model dim (2048)
    rm.encode_coordinate(&sx, 1, xemb.data());
    rm.encode_coordinate(&sy, 1, yemb.data());
    std::vector<float> prompt(a);
    prompt.insert(prompt.end(), xemb.begin(), xemb.end());
    prompt.insert(prompt.end(), yemb.begin(), yemb.end());
    prompt.insert(prompt.end(), b.begin(), b.end());
    std::vector<float> hidden(D), emb(D), lv(e.model.n_vocab);
    int nt = e.step({}, lv.data(), prompt.data(), (int)(prompt.size() / D), hidden.data(), false);
    if (nt == 0) return false;   // eos (plain argmax, per official _detect_gaze)
    // one (x,y) point, include_size=False path
    float xl[1024];
    rm.decode_coordinate(hidden.data(), xl);
    gx = RegionModel::logits_to_center(xl, 1024);
    rm.encode_coordinate(&gx, 1, emb.data());
    e.step({}, nullptr, emb.data(), 1, hidden.data(), false);
    rm.decode_coordinate(hidden.data(), xl);
    gy = RegionModel::logits_to_center(xl, 1024);
    return true;
}

// ---- minimal HTTP server ----
struct MdHttpReq {
    std::string method, path, query, body;
    std::map<std::string, std::string> headers;   // lowercased keys
};

static bool md_recv_all(SOCKET s, MdHttpReq & req) {
    std::string buf;
    char tmp[8192];
    size_t hdr_end = std::string::npos;
    while (hdr_end == std::string::npos) {
        int n = recv(s, tmp, sizeof(tmp), 0);
        if (n <= 0) return false;
        buf.append(tmp, n);
        hdr_end = buf.find("\r\n\r\n");
        if (buf.size() > 1 << 20) return false;
    }
    std::istringstream hs(buf.substr(0, hdr_end));
    std::string line;
    std::getline(hs, line);
    {
        size_t sp1 = line.find(' '), sp2 = line.rfind(' ');
        if (sp1 == std::string::npos || sp2 == sp1) return false;
        req.method = line.substr(0, sp1);
        std::string target = line.substr(sp1 + 1, sp2 - sp1 - 1);
        size_t q = target.find('?');
        if (q != std::string::npos) { req.query = target.substr(q + 1); target = target.substr(0, q); }
        req.path = target;
    }
    while (std::getline(hs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t c = line.find(':');
        if (c == std::string::npos) continue;
        std::string k = line.substr(0, c), v = line.substr(c + 1);
        while (!v.empty() && (v[0] == ' ')) v.erase(v.begin());
        for (auto & ch : k) ch = (char)tolower(ch);
        req.headers[k] = v;
    }
    size_t clen = 0;
    auto it = req.headers.find("content-length");
    if (it != req.headers.end()) clen = (size_t)atoll(it->second.c_str());
    req.body = buf.substr(hdr_end + 4);
    while (req.body.size() < clen) {
        int n = recv(s, tmp, sizeof(tmp), 0);
        if (n <= 0) break;
        req.body.append(tmp, n);
    }
    return true;
}

static void md_send(SOCKET s, const std::string & status, const std::string & ctype, const std::string & body) {
    std::string h = "HTTP/1.1 " + status + "\r\nContent-Type: " + ctype +
                    "\r\nContent-Length: " + std::to_string(body.size()) +
                    "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
    send(s, h.data(), (int)h.size(), 0);
    send(s, body.data(), (int)body.size(), 0);
}

static std::string md_query_param(const std::string & q, const char * key) {
    std::string pat = std::string(key) + "=";
    size_t p = q.find(pat);
    if (p == std::string::npos) return "";
    size_t e = q.find('&', p);
    std::string v = q.substr(p + pat.size(), e == std::string::npos ? std::string::npos : e - p - pat.size());
    md_url_decode(v);
    return v;
}

static int run_server(Engine & e, const char * vision_path, const char * region_path, int port,
                      const char * bind_host = "127.0.0.1") {
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) { fprintf(stderr, "WSAStartup failed\n"); return 1; }
#endif
    SOCKET srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv == INVALID_SOCKET) { fprintf(stderr, "socket failed\n"); return 1; }
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, (const char *)&one, sizeof(one));
    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = bind_host && strcmp(bind_host, "0.0.0.0") == 0
        ? htonl(INADDR_ANY)
        : inet_addr(bind_host ? bind_host : "127.0.0.1");   // default: local only
    addr.sin_port = htons((u_short)port);
    if (bind(srv, (sockaddr *)&addr, sizeof(addr)) != 0) { fprintf(stderr, "bind failed on port %d\n", port); return 1; }
    if (listen(srv, 4) != 0) { fprintf(stderr, "listen failed\n"); return 1; }

    // load the vision model up front (0.9 GB resident fits alongside the 10.3 GB
    // text weights) and run a 1-layer dummy encode: the first vision compute
    // initializes per-backend CUDA state, and letting that happen only after text
    // requests have run trips asserts inside ggml-cuda
    VisionModel vm;
    RegionModel rm;
    bool vm_loaded = vm.load(vision_path, e.backend), rm_loaded = rm.load(region_path);
    if (vm_loaded) {
        // full-depth warmup on dummy crops matching the common chunk shape (3):
        // absorbs the one-time first-graph cost (cuBLAS workspace, CUDA graph
        // capture) so the first real request doesn't pay it
        std::vector<float> dummy((size_t)3 * 3 * vm.crop * vm.crop, 0.0f);
        auto w = vm.encode_crops(dummy.data(), 3, vm.n_layers, e.backend);
        fprintf(stderr, "vision warmup: %zu values\n", w.size());
    }
    if (!vm_loaded) fprintf(stderr, "warning: vision model failed to load, /api/vision disabled\n");

    printf("SkyVision.cpp web UI: http://127.0.0.1:%d  (Ctrl+C to stop)\n", port);
    fflush(stdout);

    while (true) {
        sockaddr_in cli {};
        socklen_t cl = sizeof(cli);
        SOCKET c = accept(srv, (sockaddr *)&cli, &cl);
        if (c == INVALID_SOCKET) continue;
        MdHttpReq req;
        if (!md_recv_all(c, req)) { closesocket(c); continue; }

        if (req.method == "GET" && (req.path == "/" || req.path == "/index.html")) {
            static std::string ui_html;   // read once from web/index.html next to the exe
            if (ui_html.empty()) {
                FILE * f = fopen("web/index.html", "rb");
                if (f) {
                    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
                    ui_html.resize(sz);
                    if (fread(ui_html.data(), 1, sz, f) != (size_t)sz) ui_html.clear();
                    fclose(f);
                }
            }
            if (ui_html.empty())
                md_send(c, "404 Not Found", "text/plain", "web/index.html not found (run from the repo root)");
            else
                md_send(c, "200 OK", "text/html; charset=utf-8", ui_html);
        } else if (req.method == "POST" && req.path == "/api/ask") {
            std::string prompt = md_json_field(req.body, "prompt");
            long n = md_json_int(req.body, "n", 200);
            if (n < 1) n = 1;
            if (n > 1000) n = 1000;
            e.reset();
            md_cache.kv_valid = false;   // engine KV wiped -> image prefix must be rebuilt
            if (prompt.empty()) {
                md_send(c, "400 Bad Request", "application/json", "{\"error\":\"empty prompt\"}");
            } else {
                std::string out = md_generate_text(e, prompt, (int)n);
                md_send(c, "200 OK", "application/json", "{\"text\":\"" + md_json_escape(out) + "\"}");
            }
        } else if (req.method == "POST" && req.path == "/api/vision") {
            std::string q = md_query_param(req.query, "q");
            std::string mode = md_query_param(req.query, "mode");
            bool need_q = mode == "ask" || mode == "detect" || mode == "point";
            if (req.body.size() < 64 || (q.empty() && need_q)) {
                md_send(c, "400 Bad Request", "application/json", "{\"error\":\"need image body and ?q= (mode=caption needs no q)\"}");
                continue;
            }
            // image switch detection: same body hash -> reuse embeddings + KV prefix
            if (md_fnv(req.body) != md_cache.img_key) {
                e.reset();
                md_cache.img_key = md_fnv(req.body);
                md_cache.kv_valid = false;
                md_cache.prefix.clear();
            }
            if (!vm_loaded) {
                if (!vm.load(vision_path, e.backend)) { md_send(c, "500 ERR", "application/json", "{\"error\":\"vision model load failed\"}"); closesocket(c); continue; }
                vm_loaded = true;
            }
            int iw = 0, ih = 0, ic = 0;
            unsigned char * rgb = stbi_load_from_memory((const unsigned char *)req.body.data(),
                                                        (int)req.body.size(), &iw, &ih, &ic, 3);
            if (!rgb) { md_send(c, "400 Bad Request", "application/json", "{\"error\":\"image decode failed\"}"); closesocket(c); continue; }
            if (mode == "caption") {
                // official caption templates: [1, 32708, 2, <12492|6382|4059>, 3]
                std::string len = md_query_param(req.query, "len");
                int t4 = len == "short" ? 12492 : (len == "long" ? 4059 : 6382);
                md_ensure_img_prefill(e, vm, rgb, iw, ih);
                std::vector<int> ctoks = { 1, 32708, 2, t4, 3 };
                std::vector<int> out;
                int cur = e.step(ctoks, nullptr);
                out.push_back(cur);
                for (int i = 1; i < 256; i++) {
                    cur = e.step({ cur }, nullptr);
                    if (cur == 0) break;
                    out.push_back(cur);
                }
                std::string cap = e.tok.decode(out);
                md_send(c, "200 OK", "application/json", "{\"text\":\"" + md_json_escape(cap) + "\"}");
            } else if (mode == "detect") {
                if (!rm_loaded) { rm.load(region_path); rm_loaded = true; }
                auto boxes = md_detect(e, vm, rm, rgb, iw, ih, q);
                std::string j = "{\"boxes\":[";
                for (size_t i = 0; i < boxes.size(); i++) {
                    char b[128];
                    snprintf(b, sizeof(b), "[%g,%g,%g,%g]", boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]);
                    if (i) j += ",";
                    j += b;
                }
                j += "]}";
                md_send(c, "200 OK", "application/json", j);
            } else if (mode == "point") {
                if (!rm_loaded) { rm.load(region_path); rm_loaded = true; }
                auto pts = md_point(e, vm, rm, rgb, iw, ih, q);
                std::string j = "{\"points\":[";
                for (size_t i = 0; i < pts.size(); i++) {
                    char b[64];
                    snprintf(b, sizeof(b), "[%g,%g]", pts[i][0], pts[i][1]);
                    if (i) j += ",";
                    j += b;
                }
                j += "]}";
                md_send(c, "200 OK", "application/json", j);
            } else if (mode == "gaze") {
                if (!rm_loaded) { rm.load(region_path); rm_loaded = true; }
                double sx = atof(md_query_param(req.query, "x").c_str());
                double sy = atof(md_query_param(req.query, "y").c_str());
                if (sx < 0 || sx > 1 || sy < 0 || sy > 1) {
                    md_send(c, "400 Bad Request", "application/json", "{\"error\":\"mode=gaze needs ?x=&y= in [0,1]\"}");
                } else {
                    float gx, gy;
                    bool ok = md_gaze(e, vm, rm, rgb, iw, ih, (float)sx, (float)sy, gx, gy);
                    char b[96];
                    if (ok) snprintf(b, sizeof(b), "{\"x\":%g,\"y\":%g}", gx, gy);
                    else snprintf(b, sizeof(b), "{\"x\":null,\"y\":null}");
                    md_send(c, "200 OK", "application/json", b);
                }
            } else {
                std::string out = md_vqa(e, vm, rgb, iw, ih, q, 200);
                md_send(c, "200 OK", "application/json", "{\"text\":\"" + md_json_escape(out) + "\"}");
            }
            stbi_image_free(rgb);
        } else {
            md_send(c, "404 Not Found", "text/plain", "not found");
        }
        closesocket(c);
    }
    return 0;
}
