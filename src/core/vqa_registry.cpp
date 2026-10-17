// vqa_registry.cpp - architecture discovery + model factory.
//
// The single place in core that knows the backend list. Everything arch-
// specific that other layers used to hardcode (tokenizer pre-tokenizer mode,
// canonical checkpoint dir, fallback vocab) is a column of the table below.
// Wiring a new backend = one factory declaration in backend_factories.hpp,
// one `create_xxx()` definition in the backend, one row here.
#include "vqa.hpp"

#include "backend_factories.hpp"
#include "tokenizer.hpp"
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

// ---------------------------------------------------------------------------
// The arch table: one row per backend. `names` are the GGUF architecture
// labels (first = canonical, rest = aliases/legacy labels).
// ---------------------------------------------------------------------------
namespace {

struct ArchEntry {
    const char * names[4];         // zero-terminated list, canonical first
    Model * (*factory)();
    TokenizerMode tokenizer_mode;
    const char * default_dir;      // canonical models/<checkpoint> dir ("" = none)
    size_t vocab_target;           // fallback when the GGUF has no vocab_size
    bool needs_vision_gguf;        // moondream3: a *-vision.gguf sibling
    bool pure_vision;              // video tower: no tokenizer, no text side
};

const ArchEntry kArchs[] = {
    // The moondream3 engine lives behind the same interface via its adapter
    // (moondream_vqa_adapter.inc), so every backend is reachable from one binary.
    { { "moondream3", "moondream", nullptr }, create_moondream,
      TokenizerMode::PlainGpt2, "models/moondream3.1", 51200, /*vision_gguf*/ true, false },
    { { "videochat3", nullptr }, create_videochat3,
      TokenizerMode::Qwen2Regex, "models/VideoChat3-4B", 151936, false, false },
    // marlin = legacy GGUF label for the same qwen3.5 backbone; videojev =
    // qwen3.5 + a cosine decision head (score-only inference)
    { { "qwen3_5", "marlin", "videojev" }, create_qwen3_5,
      TokenizerMode::Qwen2Regex, "models/Qwen3.5-2B", 151936, false, false },
    { { "moss_vl", nullptr }, create_moss_vl,
      TokenizerMode::Qwen2Regex, "models/Moss-VL-Realtime-q4k", 151936, false, false },
    // SmolVLM (Idefics3): SigLIP tower + pixel-shuffle connector + SmolLM2 trunk
    { { "smolvlm", "idefics3", nullptr }, create_smolvlm,
      TokenizerMode::Gpt2Regex, "", 151936, false, false },
    { { "qwen2vl", "qwen2_5_vl", nullptr }, create_qwen2vl,
      TokenizerMode::Qwen2Regex, "", 151936, false, false },
    // video towers: labels or pooled embedding. ViViT is the same graph with a
    // cls token + learned position table, so it reuses the videomae backend.
    { { "videomae", "vivit", nullptr }, create_videomae,
      TokenizerMode::Qwen2Regex, "", 0, false, true },
};

const ArchEntry * find_arch(const std::string & arch) {
    for (const auto & e : kArchs)
        for (const char * const * n = e.names; *n; ++n)
            if (arch == *n) return &e;
    return nullptr;
}

}  // namespace

Model * vqa_create_model(const std::string & arch) {
    const ArchEntry * e = find_arch(arch);
    return e ? e->factory() : nullptr;
}

ArchInfo arch_info(const std::string & arch) {
    ArchInfo a;
    // "supported" is answered by the factory itself: constructing a backend only
    // allocates its own metadata (no weights are read), so wiring a new
    // architecture into the arch table is enough to make discovery list it.
    const ArchEntry * e = find_arch(arch);
    if (!e) return a;
    a.supported = true;
    a.needs_vision_gguf = e->needs_vision_gguf;
    a.pure_vision = e->pure_vision;
    a.needs_tokenizer = !e->pure_vision;
    return a;
}

std::string default_tokenizer_dir(const std::string & arch) {
    // qwen3.5 aliases that ship as their own checkpoints get their own dir
    if (arch == "marlin")   return "models/Marlin-2B";
    if (arch == "videojev") return "models/VideoJev-4B";
    const ArchEntry * e = find_arch(arch);
    return (e && e->default_dir[0]) ? std::string(e->default_dir) : std::string();
}

TokenizerMode tokenizer_mode_for_arch(const std::string & arch) {
    const ArchEntry * e = find_arch(arch);
    return e ? e->tokenizer_mode : TokenizerMode::Qwen2Regex;
}

size_t fallback_vocab_target(const std::string & arch) {
    const ArchEntry * e = find_arch(arch);
    return e ? e->vocab_target : 151936;
}

}  // namespace vqa
