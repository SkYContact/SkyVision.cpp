// sampler.hpp - shared token sampling for every backend.
// Greedy (temp<=0, top_k<=0), top-k and nucleus (top-p) with an xorshift RNG so
// results are reproducible from a caller-held seed.
#pragma once

#include "vqa.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace vqa {

inline float sampler_rand(uint64_t & state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (float)(((state * 0x2545F4914F6CDD1Dull) >> 11) * (1.0 / 9007199254740992.0));
}

// Greedy argmax over [0, n). `skip_lo..skip_hi` is never selected when
// skip_hi > skip_lo: moondream reserves ids 1..9 as control tokens and its
// decode/batch paths must never emit them.
inline int greedy_argmax(const float * v, int n, int skip_lo = 0, int skip_hi = 0) {
    const bool skip = skip_hi > skip_lo;
    int best = 0;
    float bv = -INFINITY;
    for (int i = 0; i < n; i++) {
        if (skip && i >= skip_lo && i <= skip_hi) continue;
        if (v[i] > bv) { bv = v[i]; best = i; }
    }
    return best;
}

// returns the sampled token id from `logits`
inline int sample_logits(const std::vector<float> & logits, const GenParams & gp, uint64_t & rng) {
    if (gp.temperature <= 0.0f && gp.top_k <= 0)
        return greedy_argmax(logits.data(), (int)logits.size());
    std::vector<std::pair<float, int>> cand(logits.size());
    for (int i = 0; i < (int)logits.size(); i++) cand[i] = {logits[i], i};
    std::sort(cand.begin(), cand.end(),
              [](const std::pair<float, int> & a, const std::pair<float, int> & b) { return a.first > b.first; });
    int keep = (int)cand.size();
    if (gp.top_k > 0) keep = std::min(keep, gp.top_k);
    const float inv_t = 1.0f / std::max(gp.temperature, 1e-4f);
    std::vector<float> p(keep);
    const float maxl = cand[0].first;
    float sum = 0.0f;
    for (int i = 0; i < keep; i++) { p[i] = expf((cand[i].first - maxl) * inv_t); sum += p[i]; }
    if (sum <= 0.0f) return cand[0].second;
    for (int i = 0; i < keep; i++) p[i] /= sum;
    if (gp.top_p < 1.0f) {
        float acc = 0.0f;
        int n = 0;
        while (n < keep && acc < gp.top_p) { acc += p[n]; n++; }
        keep = std::max(1, n);
    }
    const float r = sampler_rand(rng);
    float acc = 0.0f;
    for (int i = 0; i < keep; i++) {
        acc += p[i];
        if (r <= acc) return cand[i].second;
    }
    return cand[keep - 1].second;
}

}  // namespace vqa
