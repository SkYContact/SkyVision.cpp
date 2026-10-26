
// qwen3omni_c2w_test.cpp - standalone code2wav smoke: random codes, no thinker
#include "qwen3omni_talker.hpp"
#include "ggml-cpu.h"
#include <crtdbg.h>
#include <cstring>
#include <cstdio>
#include <random>
#include <cstdio>


int main(int argc, char ** argv) {
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_DEBUG);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
    const char * path = argc > 1 ? argv[1] : "E:/SkyVision-weights/qwen3-omni/qwen3omni-talker-q8_0.gguf";
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    qwen3omni::Talker t;
    if (!t.load(path, backend, allocr)) return 1;
        std::vector<std::vector<float>> codes(16);
// parity mode: codes from the oracle dump, wav out for comparison
    {
        const char * cdp = getenv("QOMNI_CODES_DUMP");
        char cdfn[256];
        snprintf(cdfn, sizeof(cdfn), "%s", (cdp && strcmp(cdp, "1")) ? cdp : "D:/moondream.cpp/parity/qwen3omni/c2w_codes.bin");
        FILE * f = fopen(cdfn, "rb");
        if (!f) { printf("codes.bin missing\n"); return 1; }
        int NF = 40;
        if (const char * cd = getenv("QOMNI_CODES_DUMP")) NF = -1;  // placeholder, handled below
        std::vector<int> raw;
        if (const char * cd = getenv("QOMNI_CODES_DUMP")) {
            fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
            NF = (int)(sz / (16 * 4));
            raw.resize(16 * NF);
            std::vector<float> rawf(16 * NF);   // engine dumps float codes
            if (fread(rawf.data(), 4, rawf.size(), f) != rawf.size()) return 1;
            for (size_t q = 0; q < rawf.size(); q++) raw[q] = (int)rawf[q];
            const char * outp = getenv("QOMNI_OUT_WAV");
            printf("codes file: %d frames, out %s\n", NF, outp ? outp : "(default)");
        } else {
            raw.resize(16 * 40);
            if (fread(raw.data(), 4, raw.size(), f) != raw.size()) return 1;
        }
        fclose(f);
        for (int g = 0; g < 16; g++)
            for (int n = 0; n < NF; n++) codes[g].push_back((float)raw[g * NF + n]);
    }
    qwen3omni::dump_c2w_mid = 1;
    {
        FILE * f = fopen("D:/moondream.cpp/parity/qwen3omni/ru0c2w.f32", "wb");
        if (f) { fwrite(t.dec[0].ru[0].c2.w.data(), 4, t.dec[0].ru[0].c2.w.size(), f); fclose(f); }
        f = fopen("D:/moondream.cpp/parity/qwen3omni/ru0c2b.f32", "wb");
        if (f && t.dec[0].ru[0].c2.has_bias) fwrite(t.dec[0].ru[0].c2.b.data(), 4, t.dec[0].ru[0].c2.b.size(), f);
        if (f) fclose(f);
    }
    {
        // dump engine-bound pw1_w of upsample.0.1 via a dequant get_rows graph
        ggml_tensor * w = t.up_cnx[0].pw1_w;
        struct ggml_init_params ip = { 1 << 22, NULL, true };
        ggml_context * c = ggml_init(ip);
        ggml_tensor * ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, 8);
        ggml_tensor * r = ggml_get_rows(c, ggml_cast(c, w, GGML_TYPE_F32), ids);
        ggml_cgraph * gf = ggml_new_graph(c);
        ggml_build_forward_expand(gf, r);
        ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(al, gf);
        int idv[8] = {0,1,2,3,500,1000,2000,4095};
        ggml_backend_tensor_set(ids, idv, 0, sizeof(int) * 8);
        ggml_backend_graph_compute(backend, gf);
        std::vector<float> out(ggml_nelements(r));
        ggml_backend_tensor_get(r, out.data(), 0, out.size() * 4);
        FILE * f = fopen("D:/moondream.cpp/parity/qwen3omni/pw1w_dump.f32", "wb");
        fwrite(out.data(), 4, out.size(), f);
        fclose(f);
        ggml_gallocr_free(al); ggml_free(c);
    }
    {
        FILE * f = fopen("D:/moondream.cpp/parity/qwen3omni/up_tc0w.f32", "wb");
        if (f) { fwrite(t.up_tc[0].w.data(), 4, t.up_tc[0].w.size(), f); fclose(f); }
    }   // also write embed + transformer out
    {
        // isolated kernel check: run the engine's ru0 c2 on the dumped s2
        std::vector<float> s2in((size_t)768 * 1272);
        FILE * f = fopen("D:/moondream.cpp/parity/qwen3omni/c2w_ru0s2.f32", "rb");
        if (f) { fread(s2in.data(), 4, s2in.size(), f); fclose(f); }
        std::vector<float> c2o((size_t)768 * 1272);
        t.run_c2_kernel(s2in, c2o);
        f = fopen("D:/moondream.cpp/parity/qwen3omni/c2kernel.f32", "wb");
        fwrite(c2o.data(), 4, c2o.size(), f); fclose(f);
    }
    std::vector<float> wav;
    if (!qwen3omni::code2wav(&t, codes, wav)) { printf("c2w failed\n"); return 1; }
    {
        FILE * f = fopen("D:/moondream.cpp/parity/qwen3omni/c2w_wav.f32", "wb");
        fwrite(wav.data(), 4, wav.size(), f);
        fclose(f);
    }
    double rms = 0; for (float v : wav) rms += (double)v * v;
    printf("wav %zu samples rms %.4f min %.3f max %.3f\n", wav.size(),
           sqrt(rms / (wav.size() ? wav.size() : 1)),
           wav.empty() ? 0.f : *std::min_element(wav.begin(), wav.end()),
           wav.empty() ? 0.f : *std::max_element(wav.begin(), wav.end()));
    return 0;
}
