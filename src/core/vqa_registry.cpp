// vqa_registry.cpp - architecture discovery + model factory.
#include "vqa.hpp"

#include "vqa_util.hpp"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <sys/stat.h>

namespace vqa {

std::string architecture_of(const char * gguf_path) {
    // peek only the metadata: ggml_init with a small metadata arena, no tensor data
    struct ggml_init_params ip = { 1024ull * 1024, NULL, /*no_alloc*/ true };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) return std::string();
    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ NULL };
    struct gguf_context * gu = gguf_init_from_file(gguf_path, p);
    std::string arch;
    if (gu) {
        int64_t id = gguf_find_key(gu, "general.architecture");
        if (id >= 0 && gguf_get_kv_type(gu, id) == GGUF_TYPE_STRING)
            arch = gguf_get_val_str(gu, id);
        gguf_free(gu);
    }
    ggml_free(ctx);
    return arch;
}

static bool is_dir(const std::string & path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return false;
    return (st.st_mode & S_IFDIR) != 0;
}

std::string resolve_tokenizer_path(const std::string & arg) {
    if (arg.empty()) return std::string();
    struct stat st;
    if (stat(arg.c_str(), &st) == 0 && !(st.st_mode & S_IFDIR)) return arg;
    if (is_dir(arg)) {
        std::string cand = arg + "/tokenizer.json";
        if (stat(cand.c_str(), &st) == 0) return cand;
    }
    return std::string();
}

// Canonical models/<checkpoint> dir per arch, matching the README's model table
// layout. Without this a GGUF parked in the repo root has no sibling
// tokenizer.json, and the CLI used to fall back to whatever tokenizer.json sat
// in the working directory - the moondream one, which silently breaks every
// Qwen-family backend (no <|image_pad|>).
std::string default_tokenizer_dir(const std::string & arch) {
    if (arch == "videochat3") return "models/VideoChat3-4B";
    if (arch == "marlin")     return "models/Marlin-2B";
    if (arch == "qwen3_5")    return "models/Qwen3.5-2B";
    if (arch == "videojev")   return "models/VideoJev-4B";
    if (arch == "moss_vl")    return "models/Moss-VL-Realtime-q4k";
    if (arch == "moondream3") return "models/moondream3.1";
    return std::string();   // videomae/vivit need no tokenizer; unknown: no hint
}

}  // namespace vqa

namespace vqa {

// The moondream3 engine lives behind the same interface via its adapter
// (moondream_vqa_adapter.inc), so every backend is reachable from one binary.

Model * vqa_create_model(const std::string & arch) {
    if (arch == "videochat3") return create_videochat3();
    if (arch == "qwen3_5" || arch == "marlin") return create_qwen3_5();   // marlin = legacy GGUF label
    // VideoJev = Qwen3.5 backbone + a cosine decision head (score-only inference)
    if (arch == "videojev") return create_qwen3_5();
    if (arch == "moss_vl") return create_moss_vl();
    if (arch == "qwen2vl" || arch == "qwen2_5_vl") return create_qwen2vl();
    // SmolVLM (Idefics3): SigLIP tower + pixel-shuffle connector + SmolLM2 trunk
    if (arch == "smolvlm" || arch == "idefics3") return create_smolvlm();
    // video towers: labels or pooled embedding. ViViT is the same graph with a cls
    // token + learned position table, so it reuses the videomae backend.
    if (arch == "videomae" || arch == "vivit") return create_videomae();
    if (arch == "mage_vl") return create_mage_vl();
    if (arch == "moondream3") return create_moondream();
    return nullptr;
}

ArchInfo arch_info(const std::string & arch) {
    ArchInfo a;
    // "supported" is answered by the factory itself: constructing a backend only
    // allocates its own metadata (no weights are read), so wiring a new
    // architecture into vqa_create_model is enough to make discovery list it.
    Model * probe = vqa_create_model(arch);
    if (!probe) return a;
    delete probe;
    a.supported = true;
    if (arch == "moondream3") {
        a.needs_vision_gguf = true;   // SigLIP tower ships as a separate gguf
    } else if (arch == "videomae" || arch == "vivit") {
        a.pure_vision = true;         // classification/embedding only
        a.needs_tokenizer = false;
    }
    return a;
}

}  // namespace vqa
