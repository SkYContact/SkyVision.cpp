// minicpmo_vocoder_test.cpp - standalone parity driver for the MiniCPM-o 4.5
// token2wav vocoder (minicpmo-vocoder.gguf, arch "token2wav"). Loads ONLY the
// vocoder sibling (no 18.7 GB main GGUF) and runs the ggml graphs on fixed
// inputs, so the same-noise / mel / campplus gates iterate in seconds:
//
//   minicpmo_vocoder_test --gguf minicpmo-vocoder.gguf --codes codes.txt
//                         --out gate/voc [--rng-dir DIR] [--mel mel.f32]
//                         [--seed 0] [--euler prefix] [--wav out.wav]
//
// Writes <out>_spk/_mel/_f0/_pcm.f32 (raw f32) for the numpy gate script.
#include "minicpmo.hpp"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::string gguf_path, codes_path, out_prefix, rng_dir, mel_path, wav_path,
        euler_prefix;
    uint32_t seed = 0;
    for (int i = 1; i < argc; i++) {
        auto eq = [&](const char * f) { return i + 1 < argc && !strcmp(argv[i], f); };
        if (eq("--gguf")) gguf_path = argv[++i];
        else if (eq("--codes")) codes_path = argv[++i];
        else if (eq("--out")) out_prefix = argv[++i];
        else if (eq("--rng-dir")) rng_dir = argv[++i];
        else if (eq("--mel")) mel_path = argv[++i];
        else if (eq("--seed")) seed = (uint32_t)atoll(argv[++i]);
        else if (eq("--euler")) euler_prefix = argv[++i];
        else if (eq("--wav")) wav_path = argv[++i];
    }
    if (gguf_path.empty() || codes_path.empty() || out_prefix.empty()) {
        fprintf(stderr, "usage: minicpmo_vocoder_test --gguf F --codes F "
                        "--out PREFIX [--rng-dir D] [--mel F] [--seed N] "
                        "[--euler PREFIX] [--wav F]\n");
        return 1;
    }
    if (!rng_dir.empty()) _putenv_s("MINICPMO_VOC_RNG_DIR", rng_dir.c_str());
    if (!mel_path.empty()) _putenv_s("MINICPMO_VOC_MEL_LOAD", mel_path.c_str());
    if (!euler_prefix.empty()) _putenv_s("MINICPMO_VOC_EULER_DUMP", euler_prefix.c_str());
    _putenv_s("MINICPMO_VOC_SEED", std::to_string(seed).c_str());

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { fprintf(stderr, "cpu backend init failed\n"); return 1; }

    minicpmo::Vocoder voc;
    if (!voc.load(backend, gguf_path.c_str())) return 1;

    std::vector<int> codes;
    {
        FILE * f = fopen(codes_path.c_str(), "r");
        if (!f) { fprintf(stderr, "cannot open %s\n", codes_path.c_str()); return 1; }
        int v = 0;
        while (fscanf(f, "%d", &v) == 1) codes.push_back(v);
        fclose(f);
    }
    fprintf(stderr, "codes: %zu\n", codes.size());

    std::vector<float> pcm;
    if (!voc.run(codes, pcm)) { fprintf(stderr, "run failed\n"); return 1; }
    fprintf(stderr, "pcm: %zu samples (%.2f s)\n", pcm.size(), pcm.size() / 24000.0);

    auto dump_env = [&](const char * suf, const std::vector<float> & v) {
        std::string fp = out_prefix + suf;
        FILE * f = fopen(fp.c_str(), "wb");
        if (f) { fwrite(v.data(), 4, v.size(), f); fclose(f); }
    };
    if (const char * dp = getenv("MINICPMO_VOC_DUMP")) {
        std::string out = dp;
        auto dump = [&](const char * suf, const std::vector<float> & v) {
            std::string fp = out + suf;
            FILE * f = fopen(fp.c_str(), "wb");
            if (f) { fwrite(v.data(), 4, v.size(), f); fclose(f); }
        };
        dump("_spk.f32", voc.out_spk);
        dump("_mel.f32", voc.out_mel);
        dump("_f0.f32", voc.out_f0);
        dump("_pcm.f32", pcm);
    }
    if (!wav_path.empty()) {
        FILE * f = fopen(wav_path.c_str(), "wb");
        if (f) {
            const uint32_t nb = (uint32_t)(pcm.size() * 2), rl = 36 + nb;
            fwrite("RIFF", 1, 4, f); fwrite(&rl, 4, 1, f); fwrite("WAVE", 1, 4, f);
            fwrite("fmt ", 1, 4, f);
            const uint32_t fs = 16, sr = 24000, br = 48000;
            const uint16_t af = 1, ch = 1, bits = 16, bl = 2;
            fwrite(&fs, 4, 1, f); fwrite(&af, 2, 1, f); fwrite(&ch, 2, 1, f);
            fwrite(&sr, 4, 1, f); fwrite(&br, 4, 1, f); fwrite(&bl, 2, 1, f);
            fwrite(&bits, 2, 1, f);
            fwrite("data", 1, 4, f); fwrite(&nb, 4, 1, f);
            std::vector<int16_t> iq(pcm.size());
            for (size_t i = 0; i < pcm.size(); i++)
                iq[i] = (int16_t)std::lround(std::max(-1.0f,
                        std::min(1.0f, pcm[i])) * 32768.0f);
            fwrite(iq.data(), 2, iq.size(), f);
            fclose(f);
        }
    }
    (void)dump_env;
    return 0;
}
