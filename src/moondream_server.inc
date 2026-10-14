// moondream_server.inc — minimal embedded web UI for the moondream CLI.
// Included from src/moondream.cpp after Engine/VisionModel are available.
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

static std::string md_vqa(Engine & e, VisionModel & vm, const uint8_t * rgb, int w, int h,
                          const std::string & question, int n_gen) {
    std::vector<float> prefix;
    md_prefill_image(e, vm, rgb, w, h, prefix);
    std::vector<int> toks = { 1, 15381, 2 };
    auto q = e.tok.encode(question);
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

static std::vector<std::array<float,4>> md_detect(Engine & e, VisionModel & vm, RegionModel & rm,
                                                  const uint8_t * rgb, int w, int h,
                                                  const std::string & obj) {
    std::vector<float> prefix;
    md_prefill_image(e, vm, rgb, w, h, prefix);
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
        int bins[2] = { md_argmax(sl, 1024), md_argmax(sl + 1024, 1024) };
        float sizes[2];
        RegionModel::bins_to_size(bins, sizes);
        rm.encode_size(sizes, 1, emb.data());
        e.step({}, nullptr, emb.data(), 1, hidden.data(), false);
        boxes.push_back({{ xc - sizes[0] / 2, yc - sizes[1] / 2, xc + sizes[0] / 2, yc + sizes[1] / 2 }});
        e.step({}, lv.data(), emb.data(), 1, nullptr, false);
        next = (lv[0] >= lv[5]) ? 0 : 5;
    }
    return boxes;
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

static int run_server(Engine & e, const char * vision_path, const char * region_path, int port) {
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
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);   // local only
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
        std::vector<float> dummy((size_t)3 * vm.crop * vm.crop, 0.0f);
        auto w = vm.encode_crops(dummy.data(), 1, /*max_layer*/ 1, e.backend);
        fprintf(stderr, "vision warmup: %zu values\n", w.size());
    }
    if (!vm_loaded) fprintf(stderr, "warning: vision model failed to load, /api/vision disabled\n");

    printf("moondream.cpp web UI: http://127.0.0.1:%d  (Ctrl+C to stop)\n", port);
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
            if (prompt.empty()) {
                md_send(c, "400 Bad Request", "application/json", "{\"error\":\"empty prompt\"}");
            } else {
                std::string out = md_generate_text(e, prompt, (int)n);
                md_send(c, "200 OK", "application/json", "{\"text\":\"" + md_json_escape(out) + "\"}");
            }
        } else if (req.method == "POST" && req.path == "/api/vision") {
            std::string q = md_query_param(req.query, "q");
            std::string mode = md_query_param(req.query, "mode");
            if (req.body.size() < 64 || (q.empty() && mode != "caption")) {
                md_send(c, "400 Bad Request", "application/json", "{\"error\":\"need image body and ?q= (mode=caption needs no q)\"}");
                continue;
            }
            e.reset();   // one conversation per request (no cross-request memory)
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
                std::vector<float> pfx;
                md_prefill_image(e, vm, rgb, iw, ih, pfx);
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
