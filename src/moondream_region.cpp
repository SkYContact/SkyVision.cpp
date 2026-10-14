// moondream_region.cpp - see .hpp. GGUF loading via gguf.h, plain F32 math.
#include "moondream_region.hpp"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>

static const double MD_PI = 3.14159265358979323846;
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   include <windows.h>
#else
#   include <sys/mman.h>
#   include <fcntl.h>
#   include <unistd.h>
#endif

int md_argmax(const float * v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

float RegionModel::logits_to_center(const float * logits, int n_logits) {
    return (float)md_argmax(logits, n_logits) / n_logits;
}

void RegionModel::bins_to_size(const int * bins, float * sizes) {
    for (int i = 0; i < 2; i++)
        sizes[i] = std::pow(2.0f, (bins[i] / 1023.0f) * 10.0f - 10.0f);
}

// ---------------------------------------------------------------- load
namespace {

struct RegionMMap {
    void * ptr = nullptr;
    size_t size = 0;
    bool load(const char * path) {
#ifdef _WIN32
        HANDLE f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                               OPEN_EXISTING, 0, NULL);
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

} // namespace

bool RegionModel::load(const char * path) {
    RegionMMap mm;
    if (!mm.load(path)) {
        fprintf(stderr, "region gguf mmap failed: %s\n", path);
        return false;
    }

    struct ggml_context * wctx = nullptr;
    struct gguf_init_params ip = { /*no_alloc*/ true, /*ctx*/ &wctx };
    struct gguf_context * gu = gguf_init_from_file(path, ip);
    if (!gu) { fprintf(stderr, "region gguf init failed\n"); return false; }

    auto get_u32 = [&](const char * k, int & v) {
        int64_t id = gguf_find_key(gu, k);
        if (id >= 0) v = (int)gguf_get_val_u32(gu, id);
    };
    const char * r = "moondream3.region.";
    get_u32(std::string(std::string(r) + "dim").c_str(), dim);
    get_u32(std::string(std::string(r) + "coord_feat_dim").c_str(), coord_feat_dim);
    get_u32(std::string(std::string(r) + "coord_out_dim").c_str(), coord_out_dim);
    get_u32(std::string(std::string(r) + "size_feat_dim").c_str(), size_feat_dim);
    get_u32(std::string(std::string(r) + "size_out_dim").c_str(), size_out_dim);

    const size_t data_off = gguf_get_data_offset(gu);
    const int64_t nt = gguf_get_n_tensors(gu);

    // map ggml tensor (from the no_alloc ctx) -> F32 host vector
    bool ok = true;
    auto fetch = [&](const char * name, std::vector<float> & dst) {
        struct ggml_tensor * t = ggml_get_tensor(wctx, name);
        if (!t) {
            fprintf(stderr, "region: missing tensor %s\n", name);
            ok = false;
            return;
        }
        const uint8_t * p = (const uint8_t *)mm.ptr + data_off +
                            gguf_get_tensor_offset(gu, gguf_find_tensor(gu, name));
        // ggml ne = [cols, rows] (ne[0] contiguous); our GGUF stores all
        // region tensors as 1-D F32 blobs, so row-major torch layout is kept.
        size_t n = (size_t)t->ne[0];
        for (int a = 1; a < GGML_MAX_DIMS && t->ne[a] > 1; a++) n *= (size_t)t->ne[a];
        if (t->type == GGML_TYPE_F32) {
            dst.assign((const float *)(const void *)p, (const float *)(const void *)p + n);
        } else if (t->type == GGML_TYPE_F16) {
            dst.resize(n);
            const uint16_t * q = (const uint16_t *)(const void *)p;
            for (size_t i = 0; i < n; i++) {
                uint32_t h = q[i];
                // half -> float
                uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, man = h & 0x3FF;
                uint32_t bits;
                if (exp == 0) bits = (sign << 31) | (man << 13);  // subnormal ~ 0
                else bits = (sign << 31) | ((exp + 127 - 15) << 10) | man << 13;
                std::memcpy(&dst[i], &bits, 4);
            }
        } else {
            fprintf(stderr, "region: %s has unsupported type %d\n", name, (int)t->type);
            ok = false;
        }
    };

    // find tensor index by name (gguf_find_tensor returns index or -1)
    // NOTE: we use ggml_get_tensor on wctx by name; offsets via gguf_find_tensor.
    fetch("coord_features.weight", coord_feat_w);
    fetch("coord_encoder.weight",  coord_enc_w);
    fetch("coord_encoder.bias",    coord_enc_b);
    fetch("coord_decoder.weight",  coord_dec_w);
    fetch("coord_decoder.bias",    coord_dec_b);
    fetch("size_features.weight",  size_feat_w);
    fetch("size_encoder.weight",   size_enc_w);
    fetch("size_encoder.bias",     size_enc_b);
    fetch("size_decoder.weight",   size_dec_w);
    fetch("size_decoder.bias",     size_dec_b);
    fetch("ln.weight",             ln_w);
    fetch("ln.bias",               ln_b);

    gguf_free(gu);
    if (wctx) ggml_free(wctx);
    mm.release();
    return ok;
}

// ---------------------------------------------------------------- math
namespace {

// dst[j] = sum_i x[i] * w[j*in + i] + b[j]   (w: torch (out,in) row-major)
inline void linear(const float * x, const float * w, const float * b,
                   int in, int out, float * dst) {
    for (int j = 0; j < out; j++) {
        float s = b ? b[j] : 0.0f;
        const float * wr = w + (size_t)j * in;
        for (int i = 0; i < in; i++) s += x[i] * wr[i];
        dst[j] = s;
    }
}

// F.layer_norm(x, (dim,), w, b), eps 1e-5 (torch default)
inline void layer_norm(const float * x, const float * w, const float * b,
                       int n, float * dst) {
    double mean = 0.0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    double var = 0.0;
    for (int i = 0; i < n; i++) { double d = x[i] - mean; var += d * d; }
    var /= n;
    double inv = 1.0 / std::sqrt(var + 1e-5);
    for (int i = 0; i < n; i++) dst[i] = (float)(((x[i] - mean) * inv) * w[i] + b[i]);
}

} // namespace

void RegionModel::encode_coordinate(const float * coords, int n, float * out) const {
    const int nf = coord_feat_dim / 2;   // 128
    std::vector<float> f(nf), feat(coord_feat_dim);
    for (int k = 0; k < n; k++) {
        for (int j = 0; j < nf; j++)
            f[j] = (float)(2.0 * MD_PI) * coords[k] * coord_feat_w[j];
        for (int j = 0; j < nf; j++) {
            feat[j] = std::cos(f[j]);
            feat[nf + j] = std::sin(f[j]);
        }
        linear(feat.data(), coord_enc_w.data(), coord_enc_b.data(),
               coord_feat_dim, dim, out + (size_t)k * dim);
    }
}

void RegionModel::decode_coordinate(const float * hidden, float * logits) const {
    std::vector<float> h(dim);
    layer_norm(hidden, ln_w.data(), ln_b.data(), dim, h.data());
    linear(h.data(), coord_dec_w.data(), coord_dec_b.data(), dim, coord_out_dim, logits);
}

void RegionModel::encode_size(const float * sizes, int n, float * out) const {
    const int nf = size_feat_dim / 2;    // 256
    std::vector<float> f(nf), feat(size_feat_dim);
    for (int k = 0; k < n; k++) {
        for (int j = 0; j < nf; j++)
            f[j] = (float)(2.0 * MD_PI) *
                   (sizes[k * 2] * size_feat_w[j] +
                    sizes[k * 2 + 1] * size_feat_w[nf + j]);
        for (int j = 0; j < nf; j++) {
            feat[j] = std::cos(f[j]);
            feat[nf + j] = std::sin(f[j]);
        }
        linear(feat.data(), size_enc_w.data(), size_enc_b.data(),
               size_feat_dim, dim, out + (size_t)k * dim);
    }
}

void RegionModel::decode_size(const float * hidden, float * logits) const {
    std::vector<float> h(dim);
    layer_norm(hidden, ln_w.data(), ln_b.data(), dim, h.data());
    linear(h.data(), size_dec_w.data(), size_dec_b.data(), dim, size_out_dim, logits);
}

// ---------------------------------------------------------------- test main
#ifdef REGION_TEST_MAIN
// CLI parity harness (driven by scripts/test_region.py):
//   moondream-region-test -m region.gguf --in in.bin --out out.bin
// in.bin:  int32 n_coords, n_sizes, has_hidden;
//          n_coords float coords; n_sizes*2 float sizes; if has_hidden,
//          dim float hidden.
// out.bin: coords embeddings (n*dim), size embeddings (n*dim),
//          coord logits (1024), size logits (2048).
#include <cstdlib>
#include <string>

int main(int argc, char ** argv) {
    std::string mp, inp, outp;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m")) mp = argv[++i];
        else if (!strcmp(argv[i], "--in")) inp = argv[++i];
        else if (!strcmp(argv[i], "--out")) outp = argv[++i];
    }
    if (mp.empty() || inp.empty() || outp.empty()) {
        fprintf(stderr, "usage: %s -m region.gguf --in in.bin --out out.bin\n", argv[0]);
        return 1;
    }
    RegionModel rm;
    if (!rm.load(mp.c_str())) return 1;

    FILE * f = fopen(inp.c_str(), "rb");
    if (!f) { fprintf(stderr, "open in failed\n"); return 1; }
    int32_t hdr[3];
    if (fread(hdr, 4, 3, f) != 3) { fprintf(stderr, "read hdr failed\n"); return 1; }
    int n_coords = hdr[0], n_sizes = hdr[1], has_hidden = hdr[2];
    std::vector<float> coords(n_coords), sizes(n_sizes * 2), hidden(rm.dim);
    if (n_coords && fread(coords.data(), 4, n_coords, f) != (size_t)n_coords) return 1;
    if (n_sizes && fread(sizes.data(), 4, n_sizes * 2, f) != (size_t)(n_sizes * 2)) return 1;
    if (has_hidden && fread(hidden.data(), 4, rm.dim, f) != (size_t)rm.dim) return 1;
    fclose(f);

    std::vector<float> cemb((size_t)n_coords * rm.dim), semb((size_t)n_sizes * rm.dim);
    std::vector<float> clog(rm.coord_out_dim), slog(rm.size_out_dim);
    if (n_coords) rm.encode_coordinate(coords.data(), n_coords, cemb.data());
    if (n_sizes)  rm.encode_size(sizes.data(), n_sizes, semb.data());
    if (has_hidden) {
        rm.decode_coordinate(hidden.data(), clog.data());
        rm.decode_size(hidden.data(), slog.data());
    }

    f = fopen(outp.c_str(), "wb");
    auto wr = [&](const std::vector<float> & v) {
        if (!v.empty()) fwrite(v.data(), 4, v.size(), f);
    };
    wr(cemb); wr(semb); wr(clog); wr(slog);
    fclose(f);
    fprintf(stderr, "region: %d coords, %d sizes, hidden=%d -> %s\n",
            n_coords, n_sizes, has_hidden, outp.c_str());
    return 0;
}
#endif
