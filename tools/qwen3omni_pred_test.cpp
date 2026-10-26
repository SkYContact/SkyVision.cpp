// qwen3omni_pred_test.cpp - oracle parity for talker trunk (TkRuntime) and
// RQ code predictor (CpRuntime) against dump_talker/{pred,tk}_*.bin raw dumps
// (extracted from pred.npz / talker.npz by the parity workflow).
#include "qwen3omni_talker.hpp"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

#ifdef _WIN32
#include <crtdbg.h>
#endif
static const bool _no_assert_dialog = []{
#ifdef _WIN32
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_DEBUG);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
#endif
    return true;
}();

static std::vector<float> rd_f32(const char * p, size_t n) {
    std::vector<float> v(n);
    FILE * f = fopen(p, "rb");
    if (!f || fread(v.data(), 4, n, f) != n) { fprintf(stderr, "read fail %s\n", p); exit(1); }
    fclose(f); return v;
}
static std::vector<long long> rd_i64(const char * p, size_t n) {
    std::vector<long long> v(n);
    FILE * f = fopen(p, "rb");
    if (!f || fread(v.data(), 8, n, f) != n) { fprintf(stderr, "read fail %s\n", p); exit(1); }
    fclose(f); return v;
}
static void stats(const char * name, const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0, na = 0, nb = 0, md = 0; int am = 0;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i];
        double d = fabs((double)a[i] - b[i]); if (d > md) { md = d; }
        if (a[i] > a[am]) am = (int)i;
    }
    double corr = dot / (sqrt(na) * sqrt(nb) + 1e-30);
    printf("  %-22s corr %.6f  maxdiff %.4g  n %zu  argmax %d\n", name, corr, md, n, am);
}

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : "E:/SkyVision-weights/qwen3-omni/qwen3omni-talker-q8_0.gguf";
    const char * dir = argc > 2 ? argv[2] : "D:/moondream.cpp/parity/qwen3omni/dump_talker";
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    qwen3omni::Talker t;
    if (!t.load(path, backend, allocr)) return 1;
    char buf[512];

    // ---------- predictor: prefill [2,1024] + 14 greedy steps ----------
    {
        const int D = 1024;
        auto emb = rd_f32((snprintf(buf, sizeof buf, "%s/pred_inputs_embeds.bin", dir), buf), 2 * D);
        auto ref_lg = rd_f32((snprintf(buf, sizeof buf, "%s/pred_logits.bin", dir), buf), 15 * 2048);
        auto ref_hd = rd_f32((snprintf(buf, sizeof buf, "%s/pred_hidden_last.bin", dir), buf), 15 * D);
        std::vector<float> lg, hd;
        if (!qwen3omni::cp_parity(&t, emb.data(), lg, hd)) { printf("cp_parity failed\n"); return 1; }
        printf("[predictor]\n");
        for (int s = 0; s < 15; s++) {
            std::vector<float> a(lg.begin() + s * 2048, lg.begin() + (s + 1) * 2048);
            std::vector<float> b(ref_lg.begin() + s * 2048, ref_lg.begin() + (s + 1) * 2048);
            stats("logits step", a, b);
        }
        stats("hidden_last all", hd, ref_hd);
        FILE * f = fopen((snprintf(buf, sizeof buf, "%s/eng_pred_logits.f32", dir), buf), "wb");
        fwrite(lg.data(), 4, lg.size(), f); fclose(f);
    }

    // ---------- talker trunk: prefill 20 rows then 3 single steps ----------
    {
        const int D = 1024, T = 20;
        auto emb = rd_f32((snprintf(buf, sizeof buf, "%s/tk_inputs_embeds.bin", dir), buf), T * D);
        auto pos = rd_i64((snprintf(buf, sizeof buf, "%s/tk_position_ids.bin", dir), buf), 3 * T);
        auto ref_lg = rd_f32((snprintf(buf, sizeof buf, "%s/tk_codec_logits_prefill.bin", dir), buf), (size_t)T * 3072);
        auto ref_hd = rd_f32((snprintf(buf, sizeof buf, "%s/tk_hidden_prefill.bin", dir), buf), (size_t)T * D);
        auto s_emb = rd_f32((snprintf(buf, sizeof buf, "%s/tk_step_inputs_embeds.bin", dir), buf), 3 * D);
        auto s_lg = rd_f32((snprintf(buf, sizeof buf, "%s/tk_step_logits.bin", dir), buf), 3 * 3072);
        auto s_hd = rd_f32((snprintf(buf, sizeof buf, "%s/tk_step_hidden_last.bin", dir), buf), 3 * D);

        std::vector<int> mpos(4 * T);
        for (int i = 0; i < T; i++) {
            mpos[i] = (int)pos[i];              // t
            mpos[T + i] = (int)pos[T + i];      // h
            mpos[2 * T + i] = (int)pos[2 * T + i]; // w
            mpos[3 * T + i] = i;                // kv slot
        }
        std::vector<float> lg, hd;
        printf("[talker prefill]\n");
        if (!qwen3omni::tk_parity(&t, emb.data(), T, mpos.data(), lg, hd)) { printf("tk prefill failed\n"); return 1; }
        stats("hidden all", hd, ref_hd);
        { std::vector<float> b(ref_lg.begin() + (size_t)(T - 1) * 3072, ref_lg.begin() + (size_t)T * 3072);
          stats("logits last", lg, b); }
        printf("[talker steps]\n");
        for (int s = 0; s < 3; s++) {
            int p = T + s;
            int m1[4] = {p, p, p, p};
            if (!qwen3omni::tk_parity(&t, s_emb.data() + (size_t)s * D, 1, m1, lg, hd)) { printf("tk step failed\n"); return 1; }
            stats("step logits", lg, std::vector<float>(s_lg.begin() + (size_t)s * 3072, s_lg.begin() + (size_t)(s + 1) * 3072));
            stats("step hidden", std::vector<float>(hd.end() - D, hd.end()),
                  std::vector<float>(s_hd.begin() + (size_t)s * D, s_hd.begin() + (size_t)(s + 1) * D));
        }
    }
    return 0;
}
