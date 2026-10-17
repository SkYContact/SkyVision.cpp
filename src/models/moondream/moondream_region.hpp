// moondream_region.hpp - moondream3.1 region head (coordinate/size encoders
// and decoders, the basis of detect/point). Official code:
// convert/ref_region.py (fourier features + small linear layers, dim 2048).
//
// All tensors are tiny (largest 2048x2048), so this is plain C++ math on F32
// host memory - no ggml graphs, CPU only.
//
// Math (F32, matching ref_region.py):
//   encode_coordinate(x): f = 2*pi*x @ Wcf   Wcf: [1,128] -> f [n,128]
//                          feat = [cos f, sin f]                [n,256]
//                          out  = feat @ Wce^T + bce            [n,2048]
//   decode_coordinate(h):  logits = (Wcd @ LN(h)) + bcd        [1024]
//                          center = argmax(logits) / 1024
//   encode_size(s):        f = 2*pi*s @ Wsf  Wsf: [2,256] -> f [n,256]
//                          feat = [cos f, sin f]                [n,512]
//                          out  = feat @ Wse^T + bse            [n,2048]
//   decode_size(h):        logits = (Wsd @ LN(h)) + bsd        [2048]
//                          viewed as (w_bins[0..1023], h_bins[1024..2047])
//   bin -> size:  size = 2^((bin/1023)*10 - 10)
#pragma once

#include <cstdint>
#include <vector>

struct RegionModel {
    // RegionConfig defaults (moondream/torch/config.py); GGUF metadata may
    // override dim / coord_feat_dim / coord_out_dim / size_feat_dim /
    // size_out_dim.
    int dim = 2048;
    int coord_feat_dim = 256;   // 128 freqs * (cos+sin)
    int coord_out_dim = 1024;
    int size_feat_dim = 512;    // 256 freqs * (cos+sin)
    int size_out_dim = 2048;    // 2 * 1024 bins

    // weights, torch (out, in) row-major F32
    std::vector<float> coord_feat_w;   // [coord_feat_dim/2]        (x -> freq)
    std::vector<float> coord_enc_w;    // [dim, coord_feat_dim]
    std::vector<float> coord_enc_b;    // [dim]
    std::vector<float> coord_dec_w;    // [coord_out_dim, dim]
    std::vector<float> coord_dec_b;    // [coord_out_dim]
    std::vector<float> size_feat_w;    // [2, size_feat_dim/2]      (w,h -> freq)
    std::vector<float> size_enc_w;     // [dim, size_feat_dim]
    std::vector<float> size_enc_b;     // [dim]
    std::vector<float> size_dec_w;     // [size_out_dim, dim]
    std::vector<float> size_dec_b;     // [size_out_dim]
    std::vector<float> ln_w, ln_b;     // [dim]

    // Loads a standalone region GGUF produced by convert/convert_region.py.
    // Returns false + stderr message on failure.
    bool load(const char * gguf_path);

    // coords: n normalized values (0..1). out: n*dim embeddings (row-major).
    void encode_coordinate(const float * coords, int n, float * out) const;
    // hidden: [dim] last text hidden state. logits: [coord_out_dim=1024].
    void decode_coordinate(const float * hidden, float * logits) const;
    // sizes: n*2 (width, height). out: n*dim embeddings.
    void encode_size(const float * sizes, int n, float * out) const;
    // hidden: [dim]. logits: [size_out_dim=2048]; [0..1023] = width bins,
    // [1024..2047] = height bins.
    void decode_size(const float * hidden, float * logits) const;

    // Convenience: center = argmax(logits)/n_logits (official).
    static float logits_to_center(const float * logits, int n_logits);
    // size = 2^((bin/1023)*10 - 10), official log-scale binning.
    static void bins_to_size(const int * bins /*2*/, float * sizes /*2*/);
};

// argmax: vqa::greedy_argmax in core/sampler.hpp (shared with the decode paths)
