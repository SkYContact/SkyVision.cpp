// omni_speech.hpp - shared building blocks for omni talker backends
// (qwen25omni / qwen3omni, and the next omni model - do not hand-roll these
// again; see the qwen3omni_talker.hpp header for the full talker protocol
// contract).
//
//   1. CodecSampler - codec-token sampling with repetition penalty, a
//      suppressed-token range (HF omni generate() suppresses the trailing
//      1024 special tokens except the codec EOS), top-k + temperature +
//      top-p, and the engine-standard xorshift64 rng. This replaces the
//      per-backend copies (qwen25omni TalkerLM::sample_codec, qwen3omni
//      talker_sample/cp_sample) - the only per-model bits are the id table
//      and the generation params, which are constructor arguments.
//   2. write_wav_24k - 24 kHz 16-bit mono RIFF writer (identical headers
//      were copy-pasted in both omni models).
//   3. Speaker tables - gguf metadata convention "qwen3omni.speaker.<name>"
//      etc. is read via vqa::gguf::u32_by_prefix (gguf_util.hpp).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace vqa {
namespace omni {

struct CodecSampler {
    int n_vocab = 0;
    float temp = 0.9f;
    int top_k = 50;
    float top_p = 1.0f;
    float rep_penalty = 1.05f;
    // suppress [suppress_from, n_vocab) except `keep` (the codec EOS); pass
    // keep < 0 to suppress the whole range (qwen25omni suppresses one bos id)
    int suppress_from = -1;
    int keep = -1;
    std::vector<int> suppress_ids;   // individual ids (qwen25omni: codec_bos)

    CodecSampler(int vocab, float t, int k, float p, float rep,
                 int suppress_from_ = -1, int keep_ = -1)
        : n_vocab(vocab), temp(t), top_k(k), top_p(p), rep_penalty(rep),
          suppress_from(suppress_from_), keep(keep_) {}

    // history supplies the repetition-penalty tokens (recent codec ids)
    int sample(std::vector<float> lg, const std::vector<int> * history,
               uint64_t & rng) const {
        if (history && rep_penalty != 1.0f) {
            for (int t : *history) {
                if (t < 0 || t >= n_vocab) continue;
                float & v = lg[(size_t)t];
                v = v > 0 ? v / rep_penalty : v * rep_penalty;
            }
        }
        if (suppress_from >= 0) {
            for (int i = std::max(0, suppress_from); i < n_vocab; i++)
                if (i != keep) lg[(size_t)i] = -INFINITY;
        }
        for (int t : suppress_ids)
            if (t >= 0 && t < n_vocab) lg[(size_t)t] = -INFINITY;
        std::vector<std::pair<float, int>> cand((size_t)n_vocab);
        for (int i = 0; i < n_vocab; i++) cand[(size_t)i] = { lg[(size_t)i], i };
        std::sort(cand.begin(), cand.end(),
                  [](const std::pair<float, int> & a, const std::pair<float, int> & b) {
                      return a.first > b.first;
                  });
        int keep_k = top_k > 0 ? std::min(top_k, n_vocab) : n_vocab;
        if (temp <= 0.01f) return cand[0].second;   // greedy
        const float inv_t = 1.0f / std::max(temp, 1e-4f);
        std::vector<float> p((size_t)keep_k);
        const float mx = cand[0].first;
        float sum = 0.0f;
        for (int i = 0; i < keep_k; i++) {
            p[(size_t)i] = expf((cand[(size_t)i].first - mx) * inv_t);
            sum += p[(size_t)i];
        }
        for (int i = 0; i < keep_k; i++) p[(size_t)i] /= sum;
        if (top_p < 1.0f) {
            float acc = 0.0f;
            int n = 0;
            while (n < keep_k && acc < top_p) { acc += p[(size_t)n]; n++; }
            keep_k = std::max(1, n);
        }
        rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
        const float r = (float)(((rng * 0x2545F4914F6CDD1Dull) >> 11)
                                * (1.0 / 9007199254740992.0));
        float acc = 0.0f;
        for (int i = 0; i < keep_k; i++) {
            acc += p[(size_t)i];
            if (r <= acc) return cand[(size_t)i].second;
        }
        return cand[(size_t)keep_k - 1].second;
    }
};

// 24 kHz 16-bit mono wav; returns false only when the file cannot be written
inline bool write_wav_24k(const char * path, const std::vector<float> & pcm) {
    FILE * f = fopen(path, "wb");
    if (!f) return false;
    const uint32_t nb = (uint32_t)(pcm.size() * 2);
    const uint32_t rl = 36 + nb;
    fwrite("RIFF", 1, 4, f); fwrite(&rl, 4, 1, f); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    { const uint32_t fs = 16; fwrite(&fs, 4, 1, f);
      const uint16_t af = 1, ch = 1, bits = 16;
      const uint32_t sr = 24000, br = 48000; const uint16_t bl = 2;
      fwrite(&af, 2, 1, f); fwrite(&ch, 2, 1, f); fwrite(&sr, 4, 1, f);
      fwrite(&br, 4, 1, f); fwrite(&bl, 2, 1, f); fwrite(&bits, 2, 1, f); }
    fwrite("data", 1, 4, f); fwrite(&nb, 4, 1, f);
    std::vector<int16_t> out(pcm.size());
    for (size_t i = 0; i < pcm.size(); i++)
        out[i] = (int16_t)std::lround(std::max(-1.0f, std::min(1.0f, pcm[i])) * 32767.0f);
    fwrite(out.data(), 2, out.size(), f);
    fclose(f);
    return true;
}

}  // namespace omni
}  // namespace vqa
