// qwen3omni_audio.hpp - Qwen3-Omni audio tower (Qwen3-ASR family).
//
// Mel: whisper-style log-mel 128 bins (core/media/audio_feats, same frontend
// as qwen25omni); the filterbank comes from a sidecar file next to the GGUF
// (mel_filters_128_201.f32, dumped from WhisperFeatureExtractor - the release
// ships no tokenizer.json/mel table, so sidecars follow the tokenizer pattern).
//
// Tower (transformers 5.20.1 Qwen3OmniMoeAudioEncoder, exact op order):
//   mel [128, T] -> chunks of n_window*2 = 100 frames (zero right-pad)
//   -> per chunk: conv2d1/2/3 (k3 s2 pad1, GELU after each) over [1, 128, Lc]
//      -> 480ch x 16 mel -> conv_out Linear(480*16 -> 1280)
//   -> + sinusoid positions (restart per chunk, sin|cos concat halves)
//   -> gather valid (chunk-major) -> 32 Whisper-style pre-norm layers with
//      WINDOW attention: windows group whole chunks, up to
//      (n_window_infer / (2*n_window)) = 8 chunks per window
//   -> ln_post -> proj1 -> GELU -> proj2(2048)
//
// The conv stack runs on the host (CPU-first stage: a few [480,16,13] feature
// maps, no point stretching the graph for it); the 32 attention layers run as
// a ggml graph like the qwen25omni tower.
#pragma once

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <string>
#include <vector>

namespace qwen3omni {

struct AudLayer {
    ggml_tensor *ln1_w = nullptr, *ln1_b = nullptr;      // self_attn_layer_norm
    ggml_tensor *wq = nullptr, *bq = nullptr;
    ggml_tensor *wk = nullptr, *bk = nullptr;
    ggml_tensor *wv = nullptr, *bv = nullptr;
    ggml_tensor *wo = nullptr, *bo = nullptr;
    ggml_tensor *ln2_w = nullptr, *ln2_b = nullptr;      // final_layer_norm
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;
    ggml_tensor *fc2_w = nullptr, *fc2_b = nullptr;
};

struct AudioTower {
    // config (from the GGUF metadata)
    int n_layer = 32, n_embd = 1280, n_head = 20, n_ff = 5120, n_mel = 128;
    int n_window = 50, n_window_infer = 800, ds_hidden = 480;
    float ln_eps = 1e-5f;

    // weights (bound by the model)
    ggml_tensor *conv1_w = nullptr, *conv1_b = nullptr;   // [480, 1, 3, 3]
    ggml_tensor *conv2_w = nullptr, *conv2_b = nullptr;   // [480, 480, 3, 3]
    ggml_tensor *conv3_w = nullptr, *conv3_b = nullptr;
    ggml_tensor *conv_out_w = nullptr;                    // [1280, 480*16]
    std::vector<AudLayer> layers;
    ggml_tensor *ln_post_w = nullptr, *ln_post_b = nullptr;
    ggml_tensor *proj1_w = nullptr, *proj1_b = nullptr;
    ggml_tensor *proj2_w = nullptr, *proj2_b = nullptr;

    std::vector<float> mel_filters_host;                  // [128, 201]
    // host F32 copies of the conv stack (F16 in the GGUF; the conv runs host)
    std::vector<float> host_c1w, host_c1b, host_c2w, host_c2b, host_c3w, host_c3b, host_cow;

    // runtime
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t allocr = nullptr;
    ggml_context * actx = nullptr;
    ggml_cgraph * agf = nullptr;
    ggml_tensor * a_in = nullptr, * a_mask = nullptr, * a_feats = nullptr;
    int cached_n_tok = -1;

    // pcm (16 kHz mono) -> [2048, n_tokens] feats (chunk-major)
    bool encode(const float * pcm, int n_samples, std::vector<float> & feats,
                int * n_tokens, const char * dump_dir);
    void free_graph();
};

}  // namespace qwen3omni
