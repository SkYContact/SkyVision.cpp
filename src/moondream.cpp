// moondream.cpp — standalone ggml inference engine for moondream3.1-9B-A2B
// Milestone 1: text-only, CPU, greedy decode, single stream, batch=1 steps.
//
// Model math (from moondream/moondream3-preview official reference text.py/layers.py/rope.py):
//   x = embd(tok)
//   per layer (parallel residual, LayerNorm with bias):
//     l   = LN(x)
//     qkv = Wqkv·l + b                    (6144 = 32h·64 q + 2·32h·64 kv, no GQA)
//     tau: tq = tanh(tau_wq·gelu(qkv))    ([32, T] per-head scalar)
//          tv = tanh(tau_wv·gelu(qkv))
//          ps  = 1 + (sigmoid(alpha·log(pos+1)) - 0.5)
//          q *= tq + ps ;  v *= tv + ps
//     rope: half-rotation on first 32 of 64 dims, theta 1.5e6 (NeoX style)
//     attn: causal softmax, scale 1/sqrt(64); out = Wout·attn + b
//     mlp:  dense 0..3 : fc1(2048->8192)+b, gelu, fc2+b
//           moe  4..23 : router(2048->64)+b, top8 by logit, softmax over the top8,
//                        fused fc1 [E, 2I, H]: h = rows[0:I], g = rows[I:2I],
//                        act = gelu(h)·(g+1), out = fc2·act, weighted sum over k
//   x = x + attn_out + mlp_out
//   logits = lm_head(LN(x)) + b

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_CUDA
#   include "ggml-cuda.h"
#endif
#include "ggml-alloc.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <utility>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "moondream_vision.hpp"
#include "moondream_preprocess.hpp"
#include "moondream_region.hpp"

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   define NOMINMAX
#   include <windows.h>
#   include <shellapi.h>
#else
#   include <sys/mman.h>
#   include <fcntl.h>
#   include <unistd.h>
#endif

// ---------------------------------------------------------------- mmap
struct MMap {
    void * ptr = nullptr;
    size_t size = 0;
    bool load(const char * path) {
#ifdef _WIN32
        HANDLE f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
        if (!f) return false;
        LARGE_INTEGER sz; GetFileSizeEx(f, &sz); size = (size_t)sz.QuadPart;
        HANDLE m = CreateFileMappingA(f, NULL, PAGE_READONLY, 0, 0, NULL);
        CloseHandle(f);
        if (!m) return false;
        ptr = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
        CloseHandle(m);
        return ptr != nullptr;
#else
        int fd = open(path, O_RDONLY);
        if (fd < 0) return false;
        size = lseek(fd, 0, SEEK_END);
        ptr = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        return ptr != MAP_FAILED;
#endif
    }
    void release() {
        if (!ptr) return;
#ifdef _WIN32
        UnmapViewOfFile(ptr);
#else
        munmap(ptr, size);
#endif
        ptr = nullptr;
    }
};

// ---------------------------------------------------------------- tokenizer
struct Tokenizer {
    std::unordered_map<std::string, int> vocab;
    std::vector<std::string> id_to_piece;
    std::map<std::pair<std::string,std::string>, int> merge_rank;   // BPE merges

    bool load(const char * path);
    std::vector<int> encode(const std::string & text) const;
    std::string decode(const std::vector<int> & toks) const;
};

// GPT-2 style bytes<->unicode mapping (vocab pieces use it)
static std::vector<int> g_byte_to_cp;      // byte -> codepoint (idx+256 for non-printables)
static std::vector<int> g_cp_to_byte;      // 256-entry map for cp in [33..323]

static void init_byte_table() {
    g_byte_to_cp.assign(256, 0);
    std::vector<int> cs;
    for (int b = 0; b < 256; b++) {
        if ((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF))
            { g_byte_to_cp[b] = b; cs.push_back(b); }
    }
    // mark used
    std::vector<bool> used(256, false);
    for (int b : cs) used[b] = true;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (!used[b]) { g_byte_to_cp[b] = 256 + n; cs.push_back(b); n++; }
    }
    g_cp_to_byte.assign(256 + n, -1);
    for (int b = 0; b < 256; b++) g_cp_to_byte[g_byte_to_cp[b]] = b;
}

static void append_utf8(std::string & out, uint32_t cp) {
    if (cp < 0x80) out += (char)cp;
    else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
    else { out += (char)(0xF0 | (cp >> 18)); out += (char)(0x80 | ((cp >> 12) & 0x3F)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
}

static uint32_t utf8_next(const unsigned char * s, size_t len, size_t & i) {
    unsigned char c = s[i];
    if (c < 0x80) { i += 1; return c; }
    if ((c & 0xE0) == 0xC0) { uint32_t v = c & 0x1F; i += 1; if (i < len) { v = (v << 6) | (s[i] & 0x3F); i += 1; } return v; }
    if ((c & 0xF0) == 0xE0) { uint32_t v = c & 0x0F; i += 1; for (int k = 0; k < 2 && i < len; k++) { v = (v << 6) | (s[i] & 0x3F); i += 1; } return v; }
    uint32_t v = c & 0x07; i += 1;
    for (int k = 0; k < 3 && i < len; k++) { v = (v << 6) | (s[i] & 0x3F); i += 1; }
    return v;
}

// parse a JSON string starting at js[i]=='"'; returns decoded content; i advanced past closing quote
static std::string json_parse_string(const std::string & js, size_t & i) {
    std::string out;
    i++;    // skip opening quote
    for (; i < js.size(); i++) {
        char c = js[i];
        if (c == '"') { i++; return out; }
        if (c != '\\') { out += c; continue; }
        char e = js[++i];
        switch (e) {
            case 'n': out += '\n'; break;
            case 't': out += '\t'; break;
            case 'r': out += '\r'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'u': {
                if (i + 4 >= js.size()) return out;
                uint32_t cp = (uint32_t)strtol(js.substr(i + 1, 4).c_str(), nullptr, 16);
                i += 4;
                if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 < js.size() && js[i + 1] == 'u') {
                    uint32_t lo = (uint32_t)strtol(js.substr(i + 3, 4).c_str(), nullptr, 16);
                    if (lo >= 0xDC00 && lo <= 0xDFFF) { cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); i += 6; }
                }
                append_utf8(out, cp);
                break;
            }
            default: out += e;
        }
    }
    return out;
}

// minimal extraction of "vocab": { piece: id, ... } from tokenizer.json
bool Tokenizer::load(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "tokenizer.json not found: %s\n", path); return false; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string js(sz, 0);
    if (fread(js.data(), 1, sz, f) != (size_t)sz) { fclose(f); return false; }
    fclose(f);

    size_t vp = js.find("\"vocab\"");
    if (vp == std::string::npos) { fprintf(stderr, "no vocab in tokenizer.json\n"); return false; }
    vp = js.find('{', vp);

    int depth = 0;
    for (size_t i = vp; i < js.size(); i++) {
        char c = js[i];
        if (c == '"') {
            std::string key = json_parse_string(js, i);
            // find ':' then number
            size_t j = js.find(':', i);
            if (j == std::string::npos) break;
            int id = atoi(js.c_str() + j + 1);
            if ((int)id_to_piece.size() <= id) id_to_piece.resize(id + 1);
            id_to_piece[id] = key;
            vocab[key] = id;
            i = j;
            continue;
        }
        if (c == '{' || c == '[') depth++;
        if (c == '}' || c == ']') { if (--depth == 0) break; }
    }
    fprintf(stderr, "tokenizer: %zu tokens\n", id_to_piece.size());

    // parse "merges": [["A","B"], ...] (nested inside "model") for proper BPE encoding
    size_t mp = js.find("\"merges\"");
    if (mp != std::string::npos) {
        mp = js.find('[', mp);
        bool have1 = false; std::string s1;
        int rank = 0, depth = 1;
        for (size_t i = mp + 1; i < js.size(); i++) {
            char c = js[i];
            if (c == '"') {
                std::string s = json_parse_string(js, i);
                i--;
                if (!have1) { s1 = s; have1 = true; }
                else { merge_rank[{s1, s}] = rank++; have1 = false; }
                continue;
            }
            if (c == '[') { depth++; continue; }
            if (c == ']') { if (--depth == 0) break; continue; }
        }
    }
    fprintf(stderr, "tokenizer: %zu merges\n", merge_rank.size());
    return !id_to_piece.empty();
}

std::string Tokenizer::decode(const std::vector<int> & toks) const {
    std::string out;
    for (int t : toks) {
        if (t < 0 || t >= (int)id_to_piece.size()) continue;
        const std::string & piece = id_to_piece[t];
        size_t i = 0;
        while (i < piece.size()) {
            uint32_t cp = utf8_next((const unsigned char *)piece.data(), piece.size(), i);
            if (cp < g_cp_to_byte.size() && g_cp_to_byte[cp] >= 0) out += (char)g_cp_to_byte[cp];
            else append_utf8(out, cp);
        }
    }
    return out;
}

std::vector<int> Tokenizer::encode(const std::string & text) const {
    // byte -> GPT-2 unicode chars
    std::vector<std::string> parts;
    for (unsigned char c : text) {
        std::string ch;
        append_utf8(ch, (uint32_t)g_byte_to_cp[c]);
        parts.push_back(ch);
    }
    // BPE: repeatedly merge the adjacent pair with the lowest merge rank
    while (parts.size() > 1) {
        int best_rank = 1 << 30; size_t best_i = 0;
        for (size_t i = 0; i + 1 < parts.size(); i++) {
            auto it = merge_rank.find({parts[i], parts[i + 1]});
            if (it != merge_rank.end() && it->second < best_rank) { best_rank = it->second; best_i = i; }
        }
        if (best_rank == (1 << 30)) break;
        parts[best_i] += parts[best_i + 1];
        parts.erase(parts.begin() + best_i + 1);
    }
    std::vector<int> out;
    for (auto & p : parts) {
        auto it = vocab.find(p);
        out.push_back(it != vocab.end() ? it->second : -1);
    }
    return out;
}

// ---------------------------------------------------------------- model
struct Layer {
    struct ggml_tensor * attn_norm_w = nullptr, * attn_norm_b = nullptr;
    struct ggml_tensor * qkv_w = nullptr, * qkv_b = nullptr;
    struct ggml_tensor * attn_out_w = nullptr, * attn_out_b = nullptr;
    struct ggml_tensor * tau_wq = nullptr, * tau_wv = nullptr, * tau_alpha = nullptr;
    struct ggml_tensor * ffn_up_w = nullptr, * ffn_up_b = nullptr;
    struct ggml_tensor * ffn_down_w = nullptr, * ffn_down_b = nullptr;
    struct ggml_tensor * router_w = nullptr, * router_b = nullptr;
    struct ggml_tensor * up_exps = nullptr, * down_exps = nullptr;
};

struct Model {
    int n_layer = 24, n_head = 32, n_kv_head = 32, n_embd = 2048, n_vocab = 51200;
    int n_head_dim = 64, n_rot = 32, n_experts = 64, n_experts_used = 8;
    int inter = 1024, ff = 8192, moe_start = 4;
    float rope_theta = 1500000.0f;

    struct ggml_context * wctx = nullptr;
    struct ggml_backend_buffer * wbuf = nullptr;
    struct ggml_context * cctx = nullptr;
    void * f32_cpu = nullptr;
    MMap mm;

    struct ggml_tensor * tok_embd = nullptr;
    struct ggml_tensor * out_norm_w = nullptr, * out_norm_b = nullptr;
    struct ggml_tensor * out_w = nullptr, * out_b = nullptr;
    std::vector<Layer> layers;
    std::vector<float> tau_alpha_host;   // host copy of all tau_alpha for per-step ps precompute

    bool load(const char * path, ggml_backend_t backend);
};

bool Model::load(const char * path, ggml_backend_t backend) {
    if (!mm.load(path)) { fprintf(stderr, "mmap failed: %s\n", path); return false; }
    fprintf(stderr, "model file: %.2f GB\n", mm.size / 1e9);

    struct gguf_context * gu = nullptr;
    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ &wctx };
    gu = gguf_init_from_file(path, p);
    if (!gu) { fprintf(stderr, "gguf init failed\n"); return false; }

    auto get_u32 = [&](const char * k, int & v) {
        int64_t id = gguf_find_key(gu, k);
        if (id >= 0) v = (int)gguf_get_val_u32(gu, id);
    };
    get_u32("moondream3.block_count", n_layer);
    get_u32("moondream3.embedding_length", n_embd);
    get_u32("moondream3.attention.head_count", n_head);
    get_u32("moondream3.attention.head_count_kv", n_kv_head);
    get_u32("moondream3.vocab_size", n_vocab);
    get_u32("moondream3.expert_count", n_experts);
    get_u32("moondream3.expert_used_count", n_experts_used);
    get_u32("moondream3.moe_start_layer", moe_start);
    {
        int64_t id = gguf_find_key(gu, "moondream3.rope.freq_base");
        if (id >= 0) rope_theta = gguf_get_val_f32(gu, id);
    }
    n_head_dim = n_embd / n_head;
    n_rot = n_head_dim / 2;

    const size_t data_off = gguf_get_data_offset(gu);
    const int64_t nt = gguf_get_n_tensors(gu);
    std::vector<char> names;
    for (int64_t i = 0; i < nt; ++i) {
        const char * name = gguf_get_tensor_name(gu, i);
        struct ggml_tensor * t = ggml_get_tensor(wctx, name);
        if (!t) continue;
        t->data = (char *)mm.ptr + data_off + gguf_get_tensor_offset(gu, i);
    }
    gguf_free(gu);

    // small F16 tensors (norms, biases) -> CPU-side F32 copies (mixed-type binary ops unsupported)
    struct ggml_backend_buffer * f32buf = nullptr;
    void * f32data = nullptr;
    size_t f32_total = 0;
    {
        struct ggml_init_params ip2 = { ggml_tensor_overhead() * 4096, NULL, true };
        cctx = ggml_init(ip2);
        size_t total = 0;
        for (auto t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
            if (t->type == GGML_TYPE_F32 || ggml_nbytes(t) > (8u << 20)) continue;
            total += ggml_nelements(t) * sizeof(float);
        }
        f32data = malloc(total);
        f32_total = total;
        size_t off = 0;
        for (auto t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
            if (t->type == GGML_TYPE_F32 || ggml_nbytes(t) > (8u << 20)) continue;
            int64_t n = ggml_nelements(t);
            const uint16_t * src = (const uint16_t *)t->data;
            float * dst = (float *)((char *)f32data + off);
            for (int64_t j = 0; j < n; j++) dst[j] = ggml_fp16_to_fp32(src[j]);
            t->data = dst;
            t->type = GGML_TYPE_F32;
            t->nb[0] = sizeof(float);
            t->nb[1] = t->ne[0] * sizeof(float);
            t->nb[2] = t->nb[1] * t->ne[1];
            t->nb[3] = t->nb[2] * t->ne[2];
            off += n * sizeof(float);
        }
        f32_cpu = f32data;   // kept for CPU zero-copy path; freed after upload on GPU path
        // host-side alpha snapshot for the per-step tau ps precompute
        tau_alpha_host.resize((size_t)n_layer * n_head);
        {
            int i = 0;
            for (auto t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
                if (strncmp(t->name, "blk.", 4) == 0 && strstr(t->name, ".tau_alpha.") != nullptr) {
                    memcpy(&tau_alpha_host[(size_t)i * n_head], t->data, sizeof(float) * n_head);
                    i++;
                }
            }
        }
        fprintf(stderr, "f32-copied small tensors: %.1f MB\n", total / 1e6);
    }

    if (ggml_backend_is_cpu(backend)) {
        // zero-copy: keep pointing weights into the mmap / f32 host memory (avoids a second
        // 10GB commit on Windows, where mmap already counts against the commit limit)
        wbuf = ggml_backend_cpu_buffer_from_ptr(mm.ptr, mm.size);
        f32buf = f32_total ? ggml_backend_cpu_buffer_from_ptr(f32data, f32_total) : nullptr;
        for (auto t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t))
            t->buffer = (t->data >= f32data && t->data < (char *)f32data + f32_total) ? f32buf : wbuf;
    } else {
        // record host-side sources, then allocate every weight on the backend and upload
        std::vector<std::pair<struct ggml_tensor *, const void *>> uploads;
        size_t total = 0;
        for (auto t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
            uploads.push_back({t, t->data});
            total += ggml_nbytes(t);
        }
        for (auto & [t, src] : uploads) t->data = nullptr;   // backend alloc only assigns data==NULL tensors
        wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
        if (!wbuf) {
            fprintf(stderr, "weight alloc on backend failed (need %.2f GB, errno %d)\n", total / 1e9, errno);
            return false;
        }
        for (auto & [t, src] : uploads)
            ggml_backend_tensor_set(t, src, 0, ggml_nbytes(t));
        fprintf(stderr, "uploaded %.2f GB weights to backend\n", total / 1e9);
        // host-side copies no longer needed; on Windows the mmap eats commit quota
        mm.release();
        free(f32_cpu); f32_cpu = nullptr;
    }

    layers.resize(n_layer);
    tok_embd   = ggml_get_tensor(wctx, "token_embd.weight");
    out_norm_w = ggml_get_tensor(wctx, "output_norm.weight");
    out_norm_b = ggml_get_tensor(wctx, "output_norm.bias");
    out_w      = ggml_get_tensor(wctx, "output.weight");
    out_b      = ggml_get_tensor(wctx, "output.bias");
    if (!tok_embd || !out_w) { fprintf(stderr, "missing core tensors\n"); return false; }

    std::vector<char> buf(256);
    auto gt = [&](const char * fmt, int i, const char * suffix) {
        snprintf(buf.data(), buf.size(), fmt, i, suffix);
        return ggml_get_tensor(wctx, buf.data());
    };
    for (int i = 0; i < n_layer; i++) {
        Layer & L = layers[i];
        L.attn_norm_w = gt("blk.%d.attn_norm.%s", i, "weight");
        L.attn_norm_b = gt("blk.%d.attn_norm.%s", i, "bias");
        L.qkv_w = gt("blk.%d.attn_qkv.%s", i, "weight");
        L.qkv_b = gt("blk.%d.attn_qkv.%s", i, "bias");
        L.attn_out_w = gt("blk.%d.attn_out.%s", i, "weight");
        L.attn_out_b = gt("blk.%d.attn_out.%s", i, "bias");
        L.tau_wq = gt("blk.%d.tau_wq.%s", i, "weight");
        L.tau_wv = gt("blk.%d.tau_wv.%s", i, "weight");
        L.tau_alpha = gt("blk.%d.tau_alpha.%s", i, "weight");
        if (i < moe_start) {
            L.ffn_up_w = gt("blk.%d.ffn_up.%s", i, "weight");
            L.ffn_up_b = gt("blk.%d.ffn_up.%s", i, "bias");
            L.ffn_down_w = gt("blk.%d.ffn_down.%s", i, "weight");
            L.ffn_down_b = gt("blk.%d.ffn_down.%s", i, "bias");
        } else {
            L.router_w = gt("blk.%d.ffn_gate_inp.%s", i, "weight");
            L.router_b = gt("blk.%d.ffn_gate_inp.%s", i, "bias");
            L.up_exps = gt("blk.%d.ffn_up_exps.%s", i, "weight");
            L.down_exps = gt("blk.%d.ffn_down_exps.%s", i, "weight");
        }
    }
    fprintf(stderr, "model loaded: %d layers, %d embd, %d vocab, experts %d/%d from layer %d\n",
            n_layer, n_embd, n_vocab, n_experts, n_experts_used, moe_start);
    return true;
}

// ---------------------------------------------------------------- engine
struct Engine {
    Model model;
    Tokenizer tok;
    int n_past = 0;
    std::vector<int *> leaked_ids;   // per-step host id buffers (freed never; few KB)
    int n_ctx = 1024;
    struct ggml_context * kv_ctx = nullptr;
    struct ggml_backend_buffer * kv_buf = nullptr;
    std::vector<struct ggml_tensor *> kv_k, kv_v;
    ggml_backend_t backend = nullptr;
    struct ggml_gallocr * gallocr = nullptr;
    // static decode graph: shape-invariant (KV views cover full n_ctx, mask hides unused
    // slots), so the same gctx/gf are reused every token — required for ggml CUDA graph
    // capture (graph key = nodes[0] pointer, must stay stable across steps)
    struct ggml_context * dec_ctx = nullptr;
    struct ggml_cgraph * dec_gf = nullptr;
    struct ggml_tensor * dec_tok_in = nullptr, * dec_pos_in = nullptr, * dec_mask_in = nullptr;
    // static batch decode graph (T = bat_n fixed, logits for all columns) — see
    // src/moondream_batch.inc
    struct ggml_context * bat_ctx = nullptr;
    struct ggml_cgraph * bat_gf = nullptr;
    struct ggml_tensor * bat_tok_in = nullptr, * bat_pos_in = nullptr, * bat_mask_in = nullptr;
    int bat_n = 0;
    int step_batch(const std::vector<int> & toks, const std::vector<int> & posv,
                   const std::vector<int> & lo, const std::vector<int> & hi, float * logits_out);

    ~Engine() {
        if (dec_ctx) ggml_free(dec_ctx);
        if (bat_ctx) ggml_free(bat_ctx);
        if (kv_buf) ggml_backend_buffer_free(kv_buf);
        if (kv_ctx) ggml_free(kv_ctx);
        if (gallocr) ggml_gallocr_free(gallocr);
        if (backend) ggml_backend_free(backend);
    }

    bool init(const char * gguf_path, const char * tok_json, int n_ctx);
    struct ggml_cgraph * build_graph(struct ggml_context * gctx, const int * toks, int n_toks,
                                     int * * host_bufs, bool fixed_ctx = false,
                                     bool emb_input = false, bool logits_all = false);   // host mem for inputs
    int step(const std::vector<int> & toks, float * last_logits_out,
             const float * emb_prefix = nullptr, int n_prefix = 0,
             float * hidden_out = nullptr, bool emb_bidir = false);
};

bool Engine::init(const char * gguf_path, const char * tok_json, int n_ctx_) {
#ifdef GGML_USE_CUDA
    if (!getenv("MD_CPU")) {
        backend = ggml_backend_cuda_init(0);
        if (backend) {
            size_t free_mm = 0, total_mm = 0;
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            ggml_backend_dev_memory(dev, &free_mm, &total_mm);
            fprintf(stderr, "backend: CUDA (%s), free %.2f GB / %.2f GB\n",
                    ggml_backend_name(backend), free_mm / 1e9, total_mm / 1e9);
        }
    }
#endif
    if (!backend) {
        backend = ggml_backend_cpu_init();
        int nth = std::max(1u, std::thread::hardware_concurrency() / 2);   // physical cores; SMT hurts GEMV
        if (const char * e = getenv("MD_THREADS")) nth = atoi(e);          // ggml CPU default is only 4
        ggml_backend_cpu_set_n_threads(backend, nth);
        fprintf(stderr, "backend: %s (%d threads)\n", ggml_backend_name(backend), nth);
    }
    if (!model.load(gguf_path, backend)) return false;
    if (!tok.load(tok_json)) return false;
    n_ctx = n_ctx_;

    struct ggml_init_params ip = { ggml_tensor_overhead() * (model.n_layer * 2 + 8), NULL, true };
    kv_ctx = ggml_init(ip);
    int n_kv = model.n_kv_head * model.n_head_dim;
    for (int i = 0; i < model.n_layer * 2; i++) {
        // F16 KV: half the memory, and ggml flash-attn CUDA consumes F16 natively
        struct ggml_tensor * t = ggml_new_tensor_2d(kv_ctx, GGML_TYPE_F16, n_kv, n_ctx); // k,v both [KV, ctx]
        (i % 2 == 0 ? kv_k : kv_v).push_back(t);
    }
    kv_buf = ggml_backend_alloc_ctx_tensors(kv_ctx, backend);
    if (!kv_buf) { fprintf(stderr, "kv cache alloc on backend failed\n"); return false; }
    ggml_backend_buffer_clear(kv_buf, 0);   // unused cache slots must be zero: flash-attn reads
    // the full fixed-size window and NaN garbage would poison softmax even under -inf mask
    return true;
}

#define MMUL(ctx, a, b) ([&]() {     if ((a)->ne[0] != (b)->ne[0] || (a)->ne[2] != (b)->ne[2] || (a)->ne[3] != (b)->ne[3])         fprintf(stderr, "BAD mul_mat: a ne[%lld,%lld,%lld,%lld] t%d  b ne[%lld,%lld,%lld,%lld] t%d\n",             (long long)(a)->ne[0], (long long)(a)->ne[1], (long long)(a)->ne[2], (long long)(a)->ne[3], (int)(a)->type,             (long long)(b)->ne[0], (long long)(b)->ne[1], (long long)(b)->ne[2], (long long)(b)->ne[3], (int)(b)->type);     return ggml_mul_mat((ctx), (a), (b)); }())

struct ggml_cgraph * Engine::build_graph(struct ggml_context * gctx, const int * toks, int n_toks,
                                         int * * host_bufs, bool fixed_ctx, bool emb_input, bool logits_all) {
    static const bool verbose = getenv("MD_TRACE") != nullptr;
    Model & M = model;
    const int D = M.n_embd, H = M.n_head, HD = M.n_head_dim, KV = M.n_kv_head * M.n_head_dim;
    const int T = n_toks, P = n_past;

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, 16384, false);

    // graph inputs: data is bound in step() via ggml_backend_tensor_set after gallocr alloc
    struct ggml_tensor * pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
    ggml_set_name(pos, "pos_in");
    (void)toks;

    struct ggml_tensor * inp = nullptr;
    if (emb_input) {
        // vision prefix: raw embeddings [D, T] fed directly (bos + 729 image tokens)
        inp = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, D, T);
        ggml_set_name(inp, "emb_in");   // note: single name; later lookups rely on it
    } else {
        struct ggml_tensor * tok_in = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
        ggml_set_name(tok_in, "tok_in");
        inp = ggml_get_rows(gctx, M.tok_embd, tok_in);
        ggml_set_name(inp, "d_h_emb");
    }

    // one causal mask shared by ALL layers (built once; step() fills it via tensor_set —
    // per-layer copies would leave 23 of them reading uninitialized memory)
    const int S_att = fixed_ctx ? n_ctx : P + T;
    struct ggml_tensor * shared_mask = nullptr;
    if (!getenv("MD_NO_FA") && (fixed_ctx || T > 1)) {
        shared_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, S_att, T);
        ggml_set_name(shared_mask, "mask_in");
    }

    // shared constants (one graph node reused by every MoE layer)
    struct ggml_tensor * one_fill = ggml_fill(gctx, ggml_new_tensor_3d(gctx, GGML_TYPE_F32, 1, 1, 1), 1.0f);

    int il_max = M.n_layer, il_min = 0;
    { const char * e = getenv("MD_MAX_LAYER"); if (e) il_max = atoi(e); }
    { const char * e = getenv("MD_MIN_LAYER"); if (e) il_min = atoi(e); }
    for (int il = il_min; il < il_max; il++) {
        Layer & L = M.layers[il];

        struct ggml_tensor * l = ggml_norm(gctx, inp, 1e-5f);
        l = ggml_mul(gctx, l, L.attn_norm_w);
        l = ggml_add(gctx, l, L.attn_norm_b);
        if (il == 0) ggml_set_name(l, "d_l0_ln");

        if (verbose) fprintf(stderr, "[b] L%d norm ok\n", il);
        struct ggml_tensor * qkv = MMUL(gctx, L.qkv_w, l);
        qkv = ggml_add(gctx, qkv, L.qkv_b);                       // [6144, T]
        if (il == 0) ggml_set_name(qkv, "d_l0_qkv");

        if (verbose) fprintf(stderr, "[b] L%d qkv ok\n", il);
        struct ggml_tensor * g_act = ggml_gelu_erf(gctx, qkv);   // ref text.py uses exact erf gelu
        struct ggml_tensor * tok_q = ggml_tanh(gctx, MMUL(gctx, L.tau_wq, g_act)); // [H, T]
        struct ggml_tensor * tok_v = ggml_tanh(gctx, MMUL(gctx, L.tau_wv, g_act));

        // ps = sigmoid(alpha*log(pos+1)) + 0.5 depends only on weights + positions: precomputed
        // on the host in step() and injected as a graph input (saves ~12 tiny nodes per layer)
        struct ggml_tensor * ps2 = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, H, T);
        ggml_format_name(ps2, "tau_ps_%d", il);

        // per-token layout: [q (32x64) | k (32x64) | v (32x64)]; cont() because rope needs formal contiguity
        struct ggml_tensor * q = ggml_cont(gctx, ggml_view_3d(gctx, qkv, HD, H, T,
                HD * sizeof(float), qkv->nb[1], 0));
        struct ggml_tensor * k = ggml_cont(gctx, ggml_view_3d(gctx, qkv, HD, M.n_kv_head, T,
                HD * sizeof(float), qkv->nb[1], H * HD * sizeof(float)));
        struct ggml_tensor * v = ggml_cont(gctx, ggml_view_3d(gctx, qkv, HD, M.n_kv_head, T,
                HD * sizeof(float), qkv->nb[1], (H + M.n_kv_head) * HD * sizeof(float)));

        if (verbose) fprintf(stderr, "[b] L%d tau ok\n", il);
        static const bool no_tau = getenv("MD_NO_TAU");
        struct ggml_tensor * sq = nullptr, * sv = nullptr;
        if (!no_tau) {
            sq = ggml_reshape_3d(gctx, ggml_add(gctx, tok_q, ps2), 1, H, T);
        q = ggml_mul(gctx, q, sq);
            sv = ggml_reshape_3d(gctx, ggml_add(gctx, tok_v, ps2), 1, H, T);
            v = ggml_mul(gctx, v, sv);
        } else {
            v = ggml_cont(gctx, v);
        }
        if (il == 0) { ggml_set_name(q, "d_l0_q_tau"); }

        q = ggml_rope_ext(gctx, q, pos, nullptr, M.n_rot, GGML_ROPE_TYPE_NEOX, n_ctx,
                          M.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(gctx, k, pos, nullptr, M.n_rot, GGML_ROPE_TYPE_NEOX, n_ctx,
                          M.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        if (il == 0) { ggml_set_name(q, "d_l0_q_rope"); ggml_set_name(k, "d_l0_k_rope"); }

        if (verbose) fprintf(stderr, "[b] L%d rope ok\n", il);
        // flatten heads: [HD*Hkv, T]
        struct ggml_tensor * k2 = ggml_reshape_2d(gctx, k, KV, T);
        struct ggml_tensor * v2 = ggml_reshape_2d(gctx, v, KV, T);
        if (il == 0) ggml_set_name(v2, "d_l0_v2");

        // write into KV cache rows P..P+T via scatter
        ggml_build_forward_expand(gf, ggml_set_rows(gctx, kv_k[il], k2, pos));
        ggml_build_forward_expand(gf, ggml_set_rows(gctx, kv_v[il], v2, pos));

        const int S = fixed_ctx ? n_ctx : P + T;   // static decode graph: fixed-size KV views
        static const bool no_fa = getenv("MD_NO_FA") != nullptr;
        struct ggml_tensor * attn = nullptr;
        if (!no_fa) {
            // fused flash attention: q [HD,T,H], k/v [HD,S,H] views straight into tensors
            // (views are row-contiguous, no cont() copies needed)
            struct ggml_tensor * qf = ggml_view_3d(gctx, q, HD, T, H, q->nb[2], q->nb[1], 0);
            struct ggml_tensor * kf = ggml_view_3d(gctx, kv_k[il], HD, S, H,
                    KV * sizeof(ggml_fp16_t), HD * sizeof(ggml_fp16_t), 0);
            struct ggml_tensor * vf = ggml_view_3d(gctx, kv_v[il], HD, S, H,
                    KV * sizeof(ggml_fp16_t), HD * sizeof(ggml_fp16_t), 0);
            // causal mask: shared tensor (may be NULL for single-shot decode without fixed ctx)
            struct ggml_tensor * mask = shared_mask;
            struct ggml_tensor * fa = ggml_flash_attn_ext(gctx, qf, kf, vf, mask,
                    1.0f / sqrtf((float)HD), 0.0f, 0.0f);          // result ne = [HD, H, T]
            if (il == 0) ggml_set_name(fa, "d_l0_attn_out3");
            attn = ggml_reshape_2d(gctx, fa, D, T);                // contiguous [HD,H,T] == [D,T]
        } else {
        // per-head attention with explicit 2D matmuls; results concatenated along D (h-major)
        for (int h = 0; h < H; h++) {
            struct ggml_tensor * qh  = ggml_view_2d(gctx, q, HD, T, q->nb[2], h * q->nb[1]);
            struct ggml_tensor * kh  = ggml_view_2d(gctx, kv_k[il], HD, S, KV * sizeof(ggml_fp16_t), h * HD * sizeof(ggml_fp16_t));
            struct ggml_tensor * wth = ggml_cont(gctx, ggml_transpose(gctx,
                    ggml_view_2d(gctx, kv_v[il], HD, S, KV * sizeof(ggml_fp16_t), h * HD * sizeof(ggml_fp16_t))));  // [S, HD]
            struct ggml_tensor * kq = MMUL(gctx, kh, qh);              // [S, T]
            kq = ggml_scale(gctx, kq, 1.0f / sqrtf((float)HD));
            if (P > 0 || T > 1) {
                kq = ggml_diag_mask_inf(gctx, kq, P);
            }
            if (il == 0 && h == 0) ggml_set_name(kq, "d_l0_scores");
            kq = ggml_soft_max(gctx, kq);
            if (il == 0 && h == 0) ggml_set_name(kq, "d_l0_attnw");
            struct ggml_tensor * oh = MMUL(gctx, wth, kq);             // [HD, T]
            attn = attn ? ggml_concat(gctx, attn, oh, 0) : oh;
        }
        }
        if (il == 0) ggml_set_name(attn, "d_l0_attn_out");
        attn = MMUL(gctx, L.attn_out_w, attn);
        attn = ggml_add(gctx, attn, L.attn_out_b);
        if (il == 0) ggml_set_name(attn, "d_l0_attn_proj");

        struct ggml_tensor * ffn = nullptr;
        if (verbose) fprintf(stderr, "[b] L%d attn ok\n", il);
        if (il < M.moe_start) {
            ffn = MMUL(gctx, L.ffn_up_w, l);
            ffn = ggml_add(gctx, ffn, L.ffn_up_b);
            ffn = ggml_gelu(gctx, ffn);
            ffn = MMUL(gctx, L.ffn_down_w, ffn);
            ffn = ggml_add(gctx, ffn, L.ffn_down_b);
            if (il == 0) ggml_set_name(ffn, "d_l0_mlp");
        } else {
            struct ggml_tensor * rlogits = ggml_add(gctx, MMUL(gctx, L.router_w, l), L.router_b); // [E, T]
            struct ggml_tensor * sel = ggml_top_k(gctx, rlogits, M.n_experts_used);   // [K, T] I32
            struct ggml_tensor * sel_logits = ggml_get_rows(gctx,
                    ggml_reshape_3d(gctx, rlogits, 1, M.n_experts, T), sel);           // [1, K, T]
            sel_logits = ggml_reshape_2d(gctx, sel_logits, M.n_experts_used, T);
            struct ggml_tensor * w = ggml_soft_max(gctx, sel_logits);                  // [K, T]
            w = ggml_reshape_3d(gctx, w, 1, M.n_experts_used, T);

            struct ggml_tensor * cur = ggml_reshape_3d(gctx, l, D, 1, T);
            struct ggml_tensor * gu = ggml_mul_mat_id(gctx, L.up_exps, cur, sel);      // [2I, K, T]
            struct ggml_tensor * h = ggml_view_3d(gctx, gu, M.inter, M.n_experts_used, T, gu->nb[1], gu->nb[2], 0);
            struct ggml_tensor * g = ggml_view_3d(gctx, gu, M.inter, M.n_experts_used, T, gu->nb[1], gu->nb[2], M.inter * gu->nb[0]);
            cur = ggml_gelu_erf(gctx, h);              // ref layers.py MoE GeGLU uses F.gelu (erf)
            g = ggml_add(gctx, g, one_fill);           // g + 1
            cur = ggml_mul(gctx, cur, g);
            struct ggml_tensor * ff3 = ggml_mul_mat_id(gctx, L.down_exps, cur, sel);   // [D, K, T]
            // weighted K-sum: broadcast-mul router weights, permute K to ne0, one sum_rows
            // (2 compute kernels instead of 15 mul/add + 16 view nodes)
            struct ggml_tensor * wb = ggml_view_3d(gctx, w, 1, M.n_experts_used, T, w->nb[1], w->nb[2], 0);
            struct ggml_tensor * prod = ggml_mul(gctx, ff3, wb);                       // [D, K, T]
            struct ggml_tensor * pk = ggml_cont(gctx, ggml_permute(gctx, prod, 1, 0, 2, 3));  // [K, D, T]
            ffn = ggml_reshape_2d(gctx, ggml_sum_rows(gctx, pk), D, T);                // [1,D,T] -> [D,T]
        }

        if (verbose) fprintf(stderr, "[b] L%d ffn ok\n", il);
        inp = ggml_add(gctx, ggml_add(gctx, inp, attn), ffn);
        if (il == 0) ggml_set_name(inp, "d_h_L0");
    }

    struct ggml_tensor * x = inp;
    if (T > 1 && !logits_all) {   // batch graphs need logits for every column
        x = ggml_view_2d(gctx, x, D, 1, x->nb[1], (T - 1) * x->nb[1]);
    }
    {   // pre-LM-head residual stream, consumed by the region head (detect/point).
        // the view must be explicitly added to the graph — an unconsumed tensor
        // never enters it and graph lookups would silently return NULL
        struct ggml_tensor * hid = ggml_view_2d(gctx, x, D, 1, x->nb[1], 0);
        ggml_set_name(hid, "hidden");
        ggml_build_forward_expand(gf, hid);
    }
    x = ggml_norm(gctx, x, 1e-5f);
    x = ggml_mul(gctx, x, M.out_norm_w);
    x = ggml_add(gctx, x, M.out_norm_b);
    struct ggml_tensor * logits = MMUL(gctx, M.out_w, x);
    logits = ggml_add(gctx, logits, M.out_b);
    ggml_set_name(logits, "logits");

    ggml_build_forward_expand(gf, logits);
    return gf;
}

int Engine::step(const std::vector<int> & toks, float * last_logits_out,
                 const float * emb_prefix, int n_prefix, float * hidden_out, bool emb_bidir) {
    static const bool prof = getenv("MD_PROF") != nullptr;
    static double t_build = 0, t_alloc = 0, t_input = 0, t_compute = 0;
    static int n_steps = 0;
    auto tick = []() { return std::chrono::steady_clock::now(); };
    n_steps++;
    const bool emb_mode = emb_prefix != nullptr;
    const int n_seq = emb_mode ? n_prefix : (int)toks.size();

    // static (shape-invariant) graph only pays off on CUDA: it enables CUDA graph capture;
    // on CPU the full-window flash attention costs more than the saved graph rebuild
    const bool use_static = (n_seq == 1) && !emb_mode && !ggml_backend_is_cpu(backend);
    struct ggml_context * gctx = nullptr;
    struct ggml_cgraph * gf = nullptr;

    auto t0 = tick();
    if (use_static) {
        if (!dec_ctx) {   // build once: fixed n_ctx KV views + always-present mask
            struct ggml_init_params ip = { 256ull * 1024 * 1024, NULL, /*no_alloc*/ true };
            dec_ctx = ggml_init(ip);
            int * hb[2] = { nullptr, nullptr };
            dec_gf = build_graph(dec_ctx, toks.data(), 1, hb, /*fixed_ctx*/ true);
            if (!gallocr) gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(gallocr, dec_gf)) { fprintf(stderr, "decode graph alloc failed\n"); return -1; }
            dec_tok_in = ggml_graph_get_tensor(dec_gf, "tok_in");
            dec_pos_in = ggml_graph_get_tensor(dec_gf, "pos_in");
            dec_mask_in = ggml_graph_get_tensor(dec_gf, "mask_in");
        }
        gctx = dec_ctx; gf = dec_gf;
    } else {
        const size_t ctx_size = 512ull * 1024 * 1024;   // graph metadata only (no_alloc + gallocr)
        struct ggml_init_params ip = { ctx_size, NULL, /*no_alloc*/ true };
        gctx = ggml_init(ip);
        if (!gctx) return -1;
        int * host_bufs[2] = { nullptr, nullptr };
        gf = build_graph(gctx, toks.data(), n_seq, host_bufs, /*fixed_ctx*/ false, /*emb_input*/ emb_mode);
        if (!gallocr) gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(gallocr, gf)) { fprintf(stderr, "graph alloc failed\n"); ggml_free(gctx); return -1; }
    }
    auto t1 = tick();

    // bind inputs
    {
        struct ggml_tensor * tok_in = use_static ? dec_tok_in : ggml_graph_get_tensor(gf, "tok_in");
        struct ggml_tensor * pos_in = use_static ? dec_pos_in : ggml_graph_get_tensor(gf, "pos_in");
        if (emb_mode) {
            // vision prefix embeddings [D, T]; memory layout d-fastest matches row-major [T][D]
            struct ggml_tensor * emb_in = ggml_graph_get_tensor(gf, "emb_in");
            ggml_backend_tensor_set(emb_in, emb_prefix, 0, sizeof(float) * (size_t)model.n_embd * n_seq);
        } else {
            ggml_backend_tensor_set(tok_in, toks.data(), 0, sizeof(int) * toks.size());
        }
        int posv0 = n_past;
        ggml_backend_tensor_set(pos_in, &posv0, 0, sizeof(int));
        if (!use_static && n_seq > 1) {
            std::vector<int> posv(n_seq);
            for (int t = 0; t < n_seq; t++) posv[t] = n_past + t;
            ggml_backend_tensor_set(pos_in, posv.data(), 0, sizeof(int) * n_seq);
        }
        struct ggml_tensor * mask = use_static ? dec_mask_in : ggml_graph_get_tensor(gf, "mask_in");
        if (mask) {   // causal F16 [S, T]: allowed when key s <= past + query t (fixed graph: also -inf on unused tail)
            const int T = n_seq;
            const int S = (int)mask->ne[0];
            std::vector<ggml_fp16_t> mv(S * T);
            for (int t = 0; t < T; t++)
                for (int s = 0; s < S; s++)
                    // the image prefix (bos + 729 crops) attends bidirectionally, like the
                    // official attn_mask[..., :730, :730] = 1; region embeddings decode causally
                    mv[t * S + s] = ggml_fp32_to_fp16((s <= n_past + t || (emb_mode && emb_bidir)) ? 0.0f : -INFINITY);
            ggml_backend_tensor_set(mask, mv.data(), 0, sizeof(ggml_fp16_t) * mv.size());
        }
        // precomputed tau position factors: ps[h] = sigmoid(alpha_il[h]*log(pos+1)) + 0.5
        {
            const int T = n_seq, H = model.n_head;
            std::vector<float> psv(H * T);
            char nm[32];
            for (int il = 0; il < model.n_layer; il++) {
                snprintf(nm, sizeof(nm), "tau_ps_%d", il);
                struct ggml_tensor * t = use_static ? ggml_graph_get_tensor(dec_gf, nm)
                                                    : ggml_graph_get_tensor(gf, nm);
                if (!t) continue;
                for (int k = 0; k < T; k++) {
                    const float p = n_past + k + 1;
                    for (int h = 0; h < H; h++)
                        psv[k * H + h] = 1.0f / (1.0f + expf(-model.tau_alpha_host[il * H + h] * logf(p))) + 0.5f;
                }
                ggml_backend_tensor_set(t, psv.data(), 0, sizeof(float) * psv.size());
            }
        }
    }
    auto t2 = tick();

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        if (!use_static) ggml_free(gctx);
        return -1;
    }
    auto t3 = tick();
    if (prof) {
        t_build  += std::chrono::duration<double>(t1 - t0).count();
        t_input  += std::chrono::duration<double>(t2 - t1).count();
        t_compute+= std::chrono::duration<double>(t3 - t2).count();
        if (n_steps % 50 == 0)
            fprintf(stderr, "[prof] steps=%d setup+input=%.2fms compute=%.2fms\n",
                    n_steps, (t_build + t_input)/n_steps*1e3, t_compute/n_steps*1e3);
    }

    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    if (hidden_out) {
        struct ggml_tensor * hid = ggml_graph_get_tensor(gf, "hidden");
        if (hid) ggml_backend_tensor_get(hid, hidden_out, 0, sizeof(float) * model.n_embd);
    }
    if (const char * ddir = getenv("MD_DUMP")) {
        const char * names[] = { "d_h_emb", "d_l0_ln", "d_l0_qkv", "d_l0_q_tau", "d_l0_q_rope", "d_l0_k_rope",
                                 "d_l0_v2", "d_l0_scores", "d_l0_attnw", "d_l0_attn_out3", "d_l0_attn_out", "d_l0_attn_proj", "d_l0_mlp", "d_h_L0" };
        std::vector<uint8_t> tmp;
        for (const char * nm : names) {
            struct ggml_tensor * t = ggml_graph_get_tensor(gf, nm);
            if (!t || !t->data) { fprintf(stderr, "dump: %s missing\n", nm); continue; }
            tmp.resize(ggml_nbytes(t));
            ggml_backend_tensor_get(t, tmp.data(), 0, ggml_nbytes(t));
            char path[512];
            snprintf(path, sizeof(path), "%s\\cpp_%s.bin", ddir, nm);
            FILE * df = fopen(path, "wb");
            int64_t n0 = t->ne[0], n1 = t->ne[1], n2 = t->ne[2];
            for (int64_t i2 = 0; i2 < n2; i2++)
                for (int64_t i1 = 0; i1 < n1; i1++)
                    fwrite(tmp.data() + i2 * t->nb[2] + i1 * t->nb[1], sizeof(float), n0, df);
            fclose(df);
        }
    }
    int argmax = 0;
    if (logits && logits->data) {
        std::vector<float> lv(model.n_vocab);
        ggml_backend_tensor_get(logits, lv.data(), 0, sizeof(float) * model.n_vocab);
        float best = -INFINITY;
        for (int i = 0; i < model.n_vocab; i++) {
            if (i >= 1 && i <= 9) continue;   // reserved control tokens never sampled
            if (lv[i] > best) { best = lv[i]; argmax = i; }
        }
        if (last_logits_out) memcpy(last_logits_out, lv.data(), sizeof(float) * model.n_vocab);
    }
    n_past += n_seq;
    if (!use_static) ggml_free(gctx);   // static decode graph stays alive for CUDA graph capture
    return argmax;
}

#include "moondream_batch.inc"   // after the Engine definition; adds step_batch + run_batch

int main(int argc, char ** argv) {
    if (getenv("MD_VERBOSE")) {
        ggml_log_set([](ggml_log_level level, const char * text, void *) {
            fprintf(stderr, "[ggml] %s", text);
        }, nullptr);
    }
#ifdef _WIN32
    // rebuild argv as UTF-8 (default ANSI argv mangles CJK prompts)
    {
        int wargc = 0;
        LPWSTR * wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
        if (wargv) {
            static std::vector<std::string> utf8_args;
            static std::vector<char *> utf8_argv;
            for (int i = 0; i < wargc; i++) {
                int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
                std::string s(n - 1, 0);
                WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, s.data(), n, nullptr, nullptr);
                utf8_args.push_back(std::move(s));
            }
            for (auto & s : utf8_args) utf8_argv.push_back(s.data());
            argv = utf8_argv.data();
            argc = wargc;
            LocalFree(wargv);
        }
    }
#endif
    const char * gguf_path = nullptr, * tok_path = nullptr, * prompt = "What color is an apple?";
    const char * img_path = nullptr, * vision_path = "moondream3.1-vision.gguf";
    const char * region_path = "moondream3.1-region.gguf";
    const char * detect_obj = nullptr, * point_obj = nullptr;
    const char * batch_path = nullptr;
    int n_gen = 32, n_ctx = 1024;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) gguf_path = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) tok_path = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) n_gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) n_ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) img_path = argv[++i];
        else if (!strcmp(argv[i], "-v") && i + 1 < argc) vision_path = argv[++i];
        else if (!strcmp(argv[i], "--detect") && i + 1 < argc) detect_obj = argv[++i];
        else if (!strcmp(argv[i], "--point") && i + 1 < argc) point_obj = argv[++i];
        else if (!strcmp(argv[i], "--region") && i + 1 < argc) region_path = argv[++i];
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch_path = argv[++i];
    }
    if (!tok_path) {
        fprintf(stderr, "usage: moondream-cli -m model.gguf -t tokenizer.json [-p prompt] [-n gen] [-c ctx] [-e text]\n");
        return 1;
    }
    init_byte_table();
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-e") && i + 1 < argc) {   // encode-only: dump prompt token ids
            const char * p = argv[++i];
            Tokenizer tz;
            if (!tz.load(tok_path)) return 1;
            std::vector<int> ids = tz.encode(p);
            for (int t : ids) printf("%d ", t);
            printf("\n");
            return 0;
        }
    }
    if (!gguf_path) {
        fprintf(stderr, "usage: moondream-cli -m model.gguf -t tokenizer.json [-p prompt] [-n gen] [-c ctx] [-e text]\n");
        return 1;
    }
    Engine e;
    if (!e.init(gguf_path, tok_path, n_ctx)) return 1;

    if (batch_path) return run_batch(e, batch_path, n_ctx, n_gen);   // moondream_batch.inc

    // optional image: encode with the vision model, prefill [bos; 729 img tokens] as raw
    // embeddings (official: inputs_embeds = cat([bos_emb, img_emb]), prefix attends bidirectionally)
    bool have_image = false;
    std::vector<float> prefix_emb;
    if (img_path) {
        int iw = 0, ih = 0, ic = 0;
        unsigned char * rgb = stbi_load(img_path, &iw, &ih, &ic, 3);
        if (!rgb) { fprintf(stderr, "image load failed: %s\n", img_path); return 1; }
        fprintf(stderr, "image: %dx%d (%s)\n", iw, ih, img_path);
        VisionModel vm;
        if (!vm.load(vision_path, e.backend)) {
            fprintf(stderr, "vision load on primary backend failed, trying CPU\n");
            ggml_backend_t cpu_be = ggml_backend_cpu_init();
            if (!vm.load(vision_path, cpu_be)) { fprintf(stderr, "vision load failed\n"); return 1; }
        }
        auto tvi = std::chrono::steady_clock::now();
        // official multi-crop Lanczos preprocessing (bit-exact vs PIL), then encoder+projection
        int th = 0, tw = 0, n_crops = 0;
        std::vector<float> crops = mdprep::prepare_crops(rgb, iw, ih, &th, &tw, &n_crops);
        stbi_image_free(rgb);
        if (crops.empty()) { fprintf(stderr, "prepare_crops failed\n"); return 1; }
        fprintf(stderr, "crops: %d (tiling %dx%d)\n", n_crops, th, tw);
        std::vector<float> enc = vm.encode_crops(crops.data(), n_crops, vm.n_layers, e.backend);
        std::vector<float> img_emb = vm.project(enc.data(), n_crops, th, tw, e.backend);   // [729, 2048]
        if (img_emb.empty()) { fprintf(stderr, "vision encode failed\n"); return 1; }
        double vsecs = std::chrono::duration<double>(std::chrono::steady_clock::now() - tvi).count();
        fprintf(stderr, "vision encoded 729 tokens in %.2fs\n", vsecs);
        // bos embedding = tok_embd row 0
        const int D = e.model.n_embd;
        std::vector<ggml_fp16_t> bos16(D);
        ggml_backend_tensor_get(e.model.tok_embd, bos16.data(), 0, sizeof(ggml_fp16_t) * D);
        prefix_emb.resize((size_t)(1 + 729) * D);
        for (int d = 0; d < D; d++) prefix_emb[d] = ggml_fp16_to_fp32(bos16[d]);
        memcpy(prefix_emb.data() + D, img_emb.data(), sizeof(float) * (size_t)729 * D);
        have_image = true;
    }

    std::vector<int> toks;
    if (have_image) {
        toks = { 1, 15381, 2 };   // query prefix; bos+image already prefilled as embeddings
    } else {
        toks = { 0, 1, 15381, 2 };   // bos + query prefix (ref_moondream.py:675)
    }
    auto q = e.tok.encode(prompt);
    toks.insert(toks.end(), q.begin(), q.end());
    toks.push_back(3);

    // ---- detect / point (official md31_moondream.py detect/_generate_points) ----
    if (have_image && (detect_obj || point_obj)) {
        RegionModel rm;
        if (!rm.load(region_path)) return 1;
        // template: detect [1,7235,476,2] / point [1,2581,2], object encoded with leading space
        std::vector<int> dtoks = detect_obj ? std::vector<int>{ 1, 7235, 476, 2 }
                                            : std::vector<int>{ 1, 2581, 2 };
        std::string obj = std::string(" ") + (detect_obj ? detect_obj : point_obj);
        auto oq = e.tok.encode(obj);
        dtoks.insert(dtoks.end(), oq.begin(), oq.end());
        dtoks.push_back(3);
        // image prefix + detect prompt prefill (image prefix step = bidirectional emb)
        e.step({}, nullptr, prefix_emb.data(), (int)(prefix_emb.size() / e.model.n_embd), nullptr, true);
        const int D = e.model.n_embd;
        std::vector<float> hidden(D), emb(D);
        std::vector<float> lv(e.model.n_vocab);
        int next = e.step(dtoks, lv.data(), nullptr, 0, hidden.data(), false);
        next = (lv[0] >= lv[5]) ? 0 : 5;   // two-way: eos_id(0) vs coord_id(5), like point_gen_indices
        if (getenv("MD_DBG")) {
            double s2 = 0; for (int i = 0; i < D; i++) s2 += (double)hidden[i] * hidden[i];
            fprintf(stderr, "DBG hidden norm=%.2f lv0=%.2f lv5=%.2f\n", s2, lv[0], lv[5]);
        }
        auto decode_emb_step = [&](const float * e2048) {
            e.step({}, nullptr, e2048, 1, hidden.data(), false);
        };
        struct Box { float x0, y0, x1, y1; };
        std::vector<Box> boxes;
        const int max_objects = 20;
        while (next != 0 && (int)boxes.size() < max_objects) {
            float xl[1024];
            rm.decode_coordinate(hidden.data(), xl);
            float xc = RegionModel::logits_to_center(xl, 1024);
            rm.encode_coordinate(&xc, 1, emb.data());
            decode_emb_step(emb.data());
            rm.decode_coordinate(hidden.data(), xl);
            float yc = RegionModel::logits_to_center(xl, 1024);
            rm.encode_coordinate(&yc, 1, emb.data());
            decode_emb_step(emb.data());
            float w = 0, h = 0;
            if (detect_obj) {
                float sl[2048];
                rm.decode_size(hidden.data(), sl);
                int bins[2] = { md_argmax(sl, 1024), md_argmax(sl + 1024, 1024) };
                float sizes[2];
                RegionModel::bins_to_size(bins, sizes);
                w = sizes[0]; h = sizes[1];
                rm.encode_size(sizes, 1, emb.data());
                decode_emb_step(emb.data());
            }
            if (detect_obj) {
                boxes.push_back({ xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2 });
            } else {
                boxes.push_back({ xc, yc, xc, yc });
            }
            // next: coord_id(5) vs eos_id(0) — official point_gen_indices two-way argmax
            e.step({}, lv.data(), emb.data(), 1, nullptr, false);
            next = (lv[0] >= lv[5]) ? 0 : 5;
        }
        if (detect_obj) {
            for (auto & b : boxes)
                printf("box [%.3f %.3f %.3f %.3f]  (x_min y_min x_max y_max, normalized)\n", b.x0, b.y0, b.x1, b.y1);
        } else {
            for (auto & b : boxes)
                printf("point (%.3f, %.3f)\n", b.x0, b.y0);
        }
        fprintf(stderr, "generated %zu object(s)\n", boxes.size());
        return 0;
    }


    fprintf(stderr, "prompt: %zu tokens:", toks.size());
    for (int t : toks) fprintf(stderr, " %d", t);
    fprintf(stderr, "\n");
    auto t0 = std::chrono::steady_clock::now();

    // streaming generation: each token is decoded and flushed as it is produced
    std::vector<int> out;
    int cur;
    if (have_image) {
        cur = e.step({}, nullptr, prefix_emb.data(), (int)(prefix_emb.size() / e.model.n_embd), nullptr, true);
        cur = e.step(toks, nullptr);
    } else {
        cur = e.step(toks, nullptr);
    }
    out.push_back(cur);
    printf("%s", e.tok.decode({ cur }).c_str());
    fflush(stdout);
    for (int i = 1; i < n_gen; i++) {
        cur = e.step({ cur }, nullptr);
        if (cur == 0) break;
        out.push_back(cur);
        printf("%s", e.tok.decode({ cur }).c_str());
        fflush(stdout);
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    printf("\n");
    fprintf(stderr, "generated %zu tokens in %.1fs (%.2f tok/s)\n", out.size() + (cur == 0 ? 1 : 0), secs, (out.size() + 1) / secs);
    return 0;
}
