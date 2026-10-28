// audio_feats.hpp - audio front-end primitives (FFT / windows / mel
// filterbanks / log-mel) shared by the audio towers (Qwen2.5-Omni whisper
// pipeline, EmbedGemma-2 Gemma4 pipeline).
//
// Same story as resample.hpp: two backends each carried a private copy of
// "FFT + Hann + mel" loops with genuinely different numerics, so both live
// here side by side instead of one borrowing from the other's directory.
//
// IMPORTANT: the two families are NOT numerically interchangeable and must
// stay that way. The whisper-family code is bit-parity with whisper.cpp /
// transformers (table-driven recursive Cooley-Tukey with a naive-DFT fallback,
// reflect padding, log10 clamped at 1e-10, max-8 global clamp, (x+4)/4). The
// Gemma4-family code matches transformers.audio_utils to 0.99999 correlation
// (iterative radix-2 FFT, float-angle periodic Hann, HTK filterbank computed
// at runtime, magnitude spectrum, log(x + floor)). Do not "unify" any float
// operation, operand width or accumulation order across the two families.
#pragma once

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>

#include <algorithm>
#include <cstdint>
#include <map>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace vqa {
namespace audio {

// ------------------------------------------------------------ whisper family
//
// Whisper.cpp's portable tables: sin/cos of 2*pi*i/n_table plus the periodic
// Hann window, precomputed once (n_table must be a multiple of every FFT size
// passed to fft_whisper; Qwen2.5-Omni uses n_table = 400, n_fft = 400).
struct WhisperTables {
    int n = 0;
    std::vector<float> sin_vals;
    std::vector<float> cos_vals;
    std::vector<float> hann_window;
};

inline const WhisperTables & whisper_tables(int n_table) {
    static std::map<int, WhisperTables> cache;
    auto it = cache.find(n_table);
    if (it == cache.end()) {
        WhisperTables & t = cache[n_table];
        t.n = n_table;
        t.sin_vals.resize(n_table);
        t.cos_vals.resize(n_table);
        t.hann_window.resize(n_table);
        for (int i = 0; i < n_table; i++) {
            const double theta = (2 * M_PI * i) / n_table;
            t.sin_vals[i] = sinf(theta);
            t.cos_vals[i] = cosf(theta);
            t.hann_window[i] = 0.5f * (1.0f - cosf((2.0f * M_PI * i) / n_table));
        }
        it = cache.find(n_table);
    }
    return it->second;
}

// Whisper's FFT: naive DFT for odd sizes, table-driven Cooley-Tukey otherwise.
// Interleaved complex output; recursion works in scratch past `in` and `out`
// (see log_mel_whisper for the buffer sizing it was written for: in has 2*N
// floats, out 8*N).
inline void fft_whisper(float * in, int N, float * out) {
    const WhisperTables & c = whisper_tables(400);  // whisper.cpp SIN_COS_N_COUNT
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }
    const int half_N = N / 2;
    if (N - half_N * 2 == 1) {
        const int sin_cos_step = 400 / N;
        for (int k = 0; k < N; k++) {
            float re = 0, im = 0;
            for (int n = 0; n < N; n++) {
                const int idx = (k * n * sin_cos_step) % 400;
                re += in[n] * c.cos_vals[idx];
                im -= in[n] * c.sin_vals[idx];
            }
            out[k * 2 + 0] = re;
            out[k * 2 + 1] = im;
        }
        return;
    }

    float * even = in + N;
    for (int i = 0; i < half_N; ++i) even[i] = in[2 * i];
    float * even_fft = out + 2 * N;
    fft_whisper(even, half_N, even_fft);

    float * odd = even;
    for (int i = 0; i < half_N; ++i) odd[i] = in[2 * i + 1];
    float * odd_fft = even_fft + N;
    fft_whisper(odd, half_N, odd_fft);

    const int sin_cos_step = 400 / N;
    for (int k = 0; k < half_N; k++) {
        const int idx = k * sin_cos_step;
        const float re = c.cos_vals[idx], im = -c.sin_vals[idx];
        const float re_odd = odd_fft[2 * k + 0], im_odd = odd_fft[2 * k + 1];
        out[2 * k + 0] = even_fft[2 * k + 0] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;
        out[2 * (k + half_N) + 0] = even_fft[2 * k + 0] - re * re_odd + im * im_odd;
        out[2 * (k + half_N) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
}

// Periodic Hann, whisper flavor: 0.5f * (1.0f - cosf(2*pi*i/n)) with the angle
// computed as double * float mix, exactly as the precomputed whisper table.
inline void hann_whisper(int n, std::vector<float> & out) {
    out.resize(n);
    for (int i = 0; i < n; i++) out[i] = 0.5f * (1.0f - cosf((2.0f * M_PI * i) / n));
}

// Whisper/HF WhisperFeatureExtractor framing + filterbank over a GGUF mel
// table (the converter dumps the exact HF matrix): reflect-pad n_fft/2 at the
// start, zero-pad past the end (torch.stft semantics), Hann window, power
// spectrum, filterbank, log10 clamped at 1e-10. `filt` is row-major
// [n_mel, n_fft/2 + 1]. Output row-major [n_mel, n_samples/hop] (assign()s,
// so `out` may come in empty). No normalization here - pair with
// normalize_log_mel_whisper.
inline void log_mel_whisper(const float * pcm, int n_samples, int n_fft, int hop, int n_mel,
                            const float * filt, std::vector<float> & out) {
    const WhisperTables & c = whisper_tables(400);
    // HF trims the centered stft to floor(L/hop) frames via the attention
    // mask downsampled ::hop - keep the extra center=True tail frame out.
    const int n_len = n_samples / hop;
    const int n_fft_bins = n_fft / 2 + 1;
    out.assign((size_t)n_mel * n_len, 0.0f);

    std::vector<float> fft_in(n_fft * 2, 0.0f);
    std::vector<float> fft_out(n_fft * 2 * 2 * 2);
    std::vector<float> power(n_fft_bins);

    const int pad = n_fft / 2;
    auto sample_at = [&](int i) {
        if (n_samples == 0) return 0.0f;
        if (i < 0) i = std::min(-i, n_samples - 1);      // reflect at the start
        if (i >= n_samples) return 0.0f;                 // torch.stft zero-pads the tail
        return pcm[i];
    };

    for (int t = 0; t < n_len; t++) {
        const int off = t * hop - pad;
        for (int j = 0; j < n_fft; j++) fft_in[j] = c.hann_window[j] * sample_at(off + j);
        fft_whisper(fft_in.data(), n_fft, fft_out.data());
        for (int j = 0; j < n_fft_bins; j++)
            power[j] = fft_out[2 * j + 0] * fft_out[2 * j + 0] +
                       fft_out[2 * j + 1] * fft_out[2 * j + 1];
        for (int m = 0; m < n_mel; m++) {
            double sum = 0.0;
            int k = 0;
            for (; k < n_fft_bins - 3; k += 4)
                sum += power[k + 0] * filt[m * n_fft_bins + k + 0] +
                       power[k + 1] * filt[m * n_fft_bins + k + 1] +
                       power[k + 2] * filt[m * n_fft_bins + k + 2] +
                       power[k + 3] * filt[m * n_fft_bins + k + 3];
            for (; k < n_fft_bins; k++) sum += power[k] * filt[m * n_fft_bins + k];
            out[(size_t)m * n_len + t] = log10(std::max(sum, 1e-10));
        }
    }
}

// Whisper log-mel normalization: global max, clamp to [max - 8, max], then
// (x + 4) / 4.
inline void normalize_log_mel_whisper(std::vector<float> & mel) {
    double mmax = -1e20;
    for (size_t i = 0; i < mel.size(); i++) mmax = std::max(mmax, (double)mel[i]);
    mmax -= 8.0;
    for (size_t i = 0; i < mel.size(); i++) {
        mel[i] = (float)std::max((double)mel[i], mmax);
        mel[i] = (mel[i] + 4.0f) / 4.0f;
    }
}

// ------------------------------------------------------------- gemma family
//
// In-place iterative radix-2 complex FFT (n must be a power of 2).
inline void fft_radix2(float * re, float * im, int n) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * (float)M_PI / (float)len;
        float wr = cosf(ang), wi = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float cr = 1.0f, ci = 0.0f;
            for (int k = 0; k < len / 2; k++) {
                int a = i + k, b = i + k + len / 2;
                float tr = re[b] * cr - im[b] * ci;
                float ti = re[b] * ci + im[b] * cr;
                re[b] = re[a] - tr; im[b] = im[a] - ti;
                re[a] += tr;        im[a] += ti;
                float nr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr; cr = nr;
            }
        }
    }
}

// Periodic Hann, Gemma4 flavor: 0.5f - 0.5f * cosf(2.0f * M_PI * n / len) with
// the whole angle computed in float. Different rounding than hann_whisper.
inline void hann_gemma(int frame_len, std::vector<float> & out) {
    out.resize(frame_len);
    for (int n = 0; n < frame_len; n++)
        out[n] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * n / frame_len);
}

// HTK mel scale, matching transformers.audio_utils (2595*log10(1 + f/700)).
inline float hz_to_mel(float f) { return 2595.0f * log10f(1.0f + f / 700.0f); }
inline float mel_to_hz(float m) { return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f); }

// HTK triangular filterbank, no norm (transformers mel_filter_bank norm=None).
// Fills fb row-major [n_bins, fft_n/2 + 1] (assign()s).
inline void mel_filterbank_htk(int n_bins, int fft_n, float min_f, float max_f, int sr,
                               std::vector<float> & fb) {
    fb.assign((size_t)n_bins * (fft_n / 2 + 1), 0.0f);
    const float m_min = hz_to_mel(min_f), m_max = hz_to_mel(max_f);
    const int n_pts = n_bins + 2;
    std::vector<float> pts(n_pts);
    for (int i = 0; i < n_pts; i++) pts[i] = mel_to_hz(m_min + (m_max - m_min) * i / (n_pts - 1));
    std::vector<float> bins(fft_n / 2 + 1);
    for (int i = 0; i <= fft_n / 2; i++) bins[i] = (float)sr * 0.5f * i / (fft_n / 2);
    for (int b = 0; b <= fft_n / 2; b++) {
        for (int j = 0; j < n_bins; j++) {
            const float lo = pts[j], mid = pts[j + 1], hi = pts[j + 2];
            float w = 0.0f;
            if (bins[b] >= lo && bins[b] <= hi) {
                if (bins[b] <= mid) w = lo == mid ? 1.0f : (bins[b] - lo) / (mid - lo);
                else                w = mid == hi ? 1.0f : (hi - bins[b]) / (hi - mid);
            }
            fb[(size_t)j * (fft_n / 2 + 1) + b] = w;
        }
    }
}

// Gemma4AudioFeatureExtractor: 16 kHz mono waveform -> log-mel [T, n_bins]
// (row-major per frame: mel[t * n_bins + j]) plus the per-frame validity mask
// (a frame is valid when its window end lands on real audio - mirrors the
// padded raw-waveform mask the HF processor builds). Semicausal left pad of
// frame_len/2, unfold window of frame_len + 1 (last sample dropped),
// magnitude spectrum, log(x + floor_). Returns false when the waveform is too
// short for one frame.
inline bool log_mel_gemma(const std::vector<float> & wav, int fft_n, int frame_len, int hop,
                          int n_bins, float floor_, float min_f, float max_f, int sr,
                          std::vector<float> & mel, std::vector<uint8_t> & valid, int & T) {
    const int pad_left = frame_len / 2;                    // semicausal time padding
    const int unfold = frame_len + 1;
    const int total = pad_left + (int)wav.size();
    const int n_frames = (total - unfold) / hop + 1;
    if (n_frames <= 0) return false;
    T = n_frames;
    mel.assign((size_t)n_frames * n_bins, 0.0f);
    valid.assign((size_t)n_frames, 0);

    // periodic Hann over frame_len
    std::vector<float> win;
    hann_gemma(frame_len, win);

    // HTK triangular filterbank, no norm (transformers mel_filter_bank norm=None)
    std::vector<float> fb;
    mel_filterbank_htk(n_bins, fft_n, min_f, max_f, sr, fb);

    std::vector<float> re(fft_n), im(fft_n);
    for (int f = 0; f < n_frames; f++) {
        std::fill(re.begin(), re.end(), 0.0f);
        std::fill(im.begin(), im.end(), 0.0f);
        const int base = f * hop;
        for (int n = 0; n < frame_len; n++) {          // unfold size 321, drop the last sample
            const int idx = base + n;
            float s = (idx >= pad_left && idx < total) ? wav[idx - pad_left] : 0.0f;
            re[n] = s * win[n];
        }
        fft_radix2(re.data(), im.data(), fft_n);
        for (int b = 0; b <= fft_n / 2; b++) {
            const float mag = sqrtf(re[b] * re[b] + im[b] * im[b]);
            for (int j = 0; j < n_bins; j++)
                mel[(size_t)f * n_bins + j] += mag * fb[(size_t)j * (fft_n / 2 + 1) + b];
        }
        // frame validity: padded-mask index i*hop + unfold - 1 must be real audio
        const int mask_idx = base + unfold - 1;
        valid[f] = (mask_idx >= pad_left && mask_idx < total) ? 1 : 0;
    }
    for (auto & v : mel) v = logf(v + floor_);
    return true;
}

// ------------------------------------------------------------- parakeet family
//
// transformers ParakeetFeatureExtractor (librosa slaney mel) family: torch
// STFT semantics (center=True, constant pad), power spectrum, librosa slaney
// filterbank computed at fp32, natural log(x + 2^-24). Numerically distinct
// from both families above (they are log10-clamped whisper / HTK-magnitude
// gemma); do not cross the wires. Used by the Nemotron-3-Omni audio tower.

// librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False, norm="slaney")
// at fp32. `fb` is row-major [n_mels, n_fft/2 + 1] (assign()s).
inline void mel_filterbank_slaney(int n_mels, int fft_n, float fmin, float fmax, int sr,
                                  std::vector<float> & fb) {
    const int n_bins = fft_n / 2 + 1;
    fb.assign((size_t)n_mels * n_bins, 0.0f);
    // slaney mel scale (all in float32, matching librosa's dtype)
    const float f_sp = 200.0f / 3.0f;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = (min_log_hz - fmin) / f_sp;
    const float logstep = logf(6.4f) / 27.0f;
    auto hz_to_mel_s = [&](float f) {
        float m = (f - fmin) / f_sp;
        if (f >= min_log_hz) m = min_log_mel + logf(f / min_log_hz) / logstep;
        return m;
    };
    auto mel_to_hz_s = [&](float m) {
        float f = fmin + f_sp * m;
        if (m >= min_log_mel) f = min_log_hz * expf(logstep * (m - min_log_mel));
        return f;
    };
    // n_mels + 2 corner frequencies (librosa linspace over the mel axis)
    const float m_lo = hz_to_mel_s(fmin), m_hi = hz_to_mel_s(fmax);
    std::vector<float> mpts((size_t)n_mels + 2), fpts((size_t)n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++) {
        const float m = m_lo + (float)i * (m_hi - m_lo) / (float)(n_mels + 1);
        mpts[(size_t)i] = m;
        fpts[(size_t)i] = mel_to_hz_s(m);
    }
    std::vector<float> bins((size_t)n_bins);
    for (int b = 0; b < n_bins; b++) bins[(size_t)b] = (float)((double)sr / 2.0 * (double)b / (double)(fft_n / 2));
    for (int m = 0; m < n_mels; m++) {
        const float fl = fpts[(size_t)m], fc = fpts[(size_t)m + 1], fh = fpts[(size_t)m + 2];
        const float enorm = 2.0f / (fh - fl);        // slaney norm
        float * row = fb.data() + (size_t)m * n_bins;
        for (int b = 0; b < n_bins; b++) {
            const float lower = (bins[(size_t)b] - fl) / (fc - fl);
            const float upper = (fh - bins[(size_t)b]) / (fh - fc);
            row[b] = std::max(0.0f, std::min(lower, upper)) * enorm;
        }
    }
}

// ParakeetFeatureExtractor.__call__ on one mono clip: preemphasis 0.97 inside
// the valid length, torch.stft(center=True, pad_mode="constant") power
// spectrum (hann periodic=False, win 400, n_fft 512, hop 160), slaney
// filterbank, log(x + 2^-24). Output [T, n_mels] row-major PER FRAME
// (mel[t * n_mels + j]); T = 1 + n_samples / hop. No CMVN here - pair with
// the caller's per-utterance CMVN over the valid frames.
inline void log_mel_parakeet(const float * pcm, int n_samples, int n_fft, int hop,
                             int win_length, int n_mels, float preemph,
                             const float * fb, std::vector<float> & mel) {
    // preemphasis (single clip: no batch padding, the timemask covers all L)
    std::vector<float> x((size_t)n_samples);
    if (n_samples > 0) x[0] = pcm[0];
    for (int i = 1; i < n_samples; i++) x[(size_t)i] = pcm[i] - preemph * pcm[i - 1];

    const int pad = n_fft / 2;
    const int T = 1 + n_samples / hop;
    mel.assign((size_t)T * n_mels, 0.0f);

    // hann periodic=False (symmetric): 0.5 - 0.5 cos(2 pi i / (n-1))
    std::vector<float> win((size_t)win_length);
    for (int i = 0; i < win_length; i++)
        win[(size_t)i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)i / (float)(win_length - 1));

    const int n_bins = n_fft / 2 + 1;
    std::vector<float> re(n_fft), im(n_fft), pw(n_bins);
    // torch.stft(win_length < n_fft): the window is CENTERED in the fft frame
    // ((n_fft - win_length)/2 zeros on each side)
    const int win_left = (n_fft - win_length) / 2;
    for (int t = 0; t < T; t++) {
        const int base = t * hop - pad;               // center=True constant pad
        std::fill(re.begin(), re.end(), 0.0f);
        std::fill(im.begin(), im.end(), 0.0f);
        for (int j = 0; j < win_length; j++) {
            const int idx = base + win_left + j;
            const float s = (idx >= 0 && idx < n_samples) ? x[(size_t)idx] : 0.0f;
            re[(size_t)(win_left + j)] = s * win[(size_t)j];
        }
        fft_radix2(re.data(), im.data(), n_fft);
        for (int b = 0; b < n_bins; b++)
            pw[(size_t)b] = re[(size_t)b] * re[(size_t)b] + im[(size_t)b] * im[(size_t)b];
        float * row = mel.data() + (size_t)t * n_mels;
        for (int m = 0; m < n_mels; m++) {
            double sum = 0.0;
            const float * frow = fb + (size_t)m * n_bins;
            for (int b = 0; b < n_bins; b++) sum += (double)frow[b] * (double)pw[(size_t)b];
            row[m] = logf((float)sum + 6.103515625e-08f);   // + 2^-24
        }
    }
}

}  // namespace audio
}  // namespace vqa
