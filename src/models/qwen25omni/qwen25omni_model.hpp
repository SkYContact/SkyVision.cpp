// qwen25omni_model.hpp - Qwen2.5-Omni backend (thinker side): Qwen2.5 text
// trunk + Qwen2.5-VL vision tower + Whisper-style audio tower, one GGUF.
// The talker GGUF (speech output) loads as an optional sibling.
#pragma once

#include "vqa.hpp"

#include "ggml-backend.h"
#include "ggml.h"

#include <string>
#include <vector>

namespace qwen25omni {

// whisper.cpp-style 128-bin log-mel. Layout [n_mel, n_len] (one row per bin).
struct MelSpec {
    int n_mel = 0;
    int n_len = 0;
    std::vector<float> data;
};

// chunked-convolution geometry shared by the graph builder and the runner:
// 200-mel-frame chunks with 4 zero columns between them.
struct AudioGeometry {
    int n_chunks = 0;
    std::vector<int> chunk_len;    // mel frames per chunk (<= 200)
    std::vector<int> n_out;        // valid conv2 columns per chunk
    int T_pad = 0;                 // conv1 columns (chunks + 4-col gaps)
    int cols = 0;                  // conv2 columns (graph width)
    int n_valid = 0, n_pool = 0;

    AudioGeometry() = default;
    AudioGeometry(int mel_len, int window2) {
        n_chunks = (mel_len + window2 - 1) / window2;
        if (n_chunks < 1) n_chunks = 1;
        chunk_len.resize((size_t)n_chunks);
        n_out.resize((size_t)n_chunks);
        for (int c = 0; c < n_chunks; c++) {
            const int rem = mel_len - c * window2;
            chunk_len[(size_t)c] = std::min(rem, window2) > 0 ? std::min(rem, window2) : 1;
            n_out[(size_t)c] = (chunk_len[(size_t)c] - 1) / 2 + 1;
        }
        T_pad = 200 * n_chunks + 4 * (n_chunks - 1);
        cols = (T_pad - 1) / 2 + 1;
        for (int c = 0; c < n_chunks; c++) {
            n_valid += n_out[(size_t)c];
            n_pool += pooled_pairs(n_out[(size_t)c]);
        }
    }

    int mel_start(int c) const { return c * (200 + 4); }   // first mel column
    int slot_start(int c) const { return c * 102; }        // first conv2 column

    // HF (v - 2) // 2 + 1 with floor division: an odd trailing frame is dropped
    static int pooled_pairs(int v) { return v < 2 ? 0 : (v - 2) / 2 + 1; }
};

// audio tower weights (pointers into the model's weight context)
struct AudLayer {
    ggml_tensor *ln1_w = nullptr, *ln1_b = nullptr;
    ggml_tensor *wq = nullptr, *bq = nullptr;
    ggml_tensor *wk = nullptr;                    // whisper encoder: k has no bias
    ggml_tensor *wv = nullptr, *bv = nullptr;
    ggml_tensor *wo = nullptr, *bo = nullptr;
    ggml_tensor *ln2_w = nullptr, *ln2_b = nullptr;
    ggml_tensor *w_gate = nullptr, *b_gate = nullptr;
    ggml_tensor *w_down = nullptr, *b_down = nullptr;
};

// Mel-spectrogram + audio-tower encoder. Weights are bound by the model before
// use; the graph is rebuilt per input (its shape depends on the mel length).
struct AudioTower {
    // preprocessing -----------------------------------------------------------
    // 16 kHz mono PCM -> log-mel [128, T] (hop 160, n_fft 400, reflect pads).
    MelSpec log_mel(const float * pcm, int n_samples) const;

    // inference ---------------------------------------------------------------
    // mel -> [n_tokens, out_dim] features in chunk-major order; n_tokens =
    // sum((len_c - 2) // 2 + 1) over conv-length chunks (25 tokens / second).
    bool encode(const MelSpec & mel, std::vector<float> & feats, int * n_tokens);

    // set by the model after loading weights
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t allocr = nullptr;
    int n_threads = 0;
    std::string dump_dir;

    // hyperparams (mirrored from the GGUF metadata)
    int n_layer = 32, n_embd = 1280, n_head = 20, n_ff = 5120;
    int n_mel = 128, n_window = 100, n_pos = 1500, out_dim = 2048;
    float ln_eps = 1e-5f;
    int hop = 160, n_fft = 400;

    ggml_tensor *conv1_w = nullptr, *conv1_b = nullptr;
    ggml_tensor *conv2_w = nullptr, *conv2_b = nullptr;
    ggml_tensor *pos_embd = nullptr;              // [n_pos, n_embd] sinusoids
    ggml_tensor *mel_filters = nullptr;           // [n_mel, n_fft/2+1] f32
    std::vector<float> mel_filters_host;          // host copy (log_mel reads it
                                                  // directly; on GPU t->data is null)
    ggml_tensor *ln_post_w = nullptr, *ln_post_b = nullptr;
    ggml_tensor *proj_w = nullptr, *proj_b = nullptr;
    std::vector<AudLayer> layers;

private:
    bool build_graph(const AudioGeometry & geo);

    ggml_context * actx = nullptr;
    ggml_cgraph * agf = nullptr;
    ggml_tensor *a_in = nullptr, *a_mask_col = nullptr, *a_pos_idx = nullptr;
    ggml_tensor *a_attn_mask = nullptr, *a_valid_idx = nullptr;
    ggml_tensor *a_pool_idx = nullptr, *a_poolr_idx = nullptr;
    ggml_tensor *a_feats = nullptr;
};

}  // namespace qwen25omni
