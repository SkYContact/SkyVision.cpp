// vqa_registry.cpp - architecture discovery + model factory.
//
// The single place in core that knows the backend list. Everything arch-
// specific that other layers used to hardcode (tokenizer pre-tokenizer mode,
// canonical checkpoint dir, fallback vocab) is a column of the table below.
// Wiring a new backend = one factory declaration in backend_factories.hpp,
// one `create_xxx()` definition in the backend, one row here.
#include "core/vqa.hpp"

#include "core/runtime/backend_factories.hpp"
#include "core/text/tokenizer.hpp"

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
    const char * web_page;         // dedicated web UI page ("" = generic vc3.html)
};

const ArchEntry kArchs[] = {
    // The moondream3 engine lives behind the same interface via its adapter
    // (moondream_vqa_adapter.inc), so every backend is reachable from one binary.
    { { "moondream3", "moondream", nullptr }, create_moondream,
      TokenizerMode::PlainGpt2, "models/moondream3.1", 51200, /*vision_gguf*/ true, false, "moondream.html" },
    { { "videochat3", nullptr }, create_videochat3,
      TokenizerMode::Qwen2Regex, "models/VideoChat3-4B", 151936, false, false, "" },
    // marlin = legacy GGUF label for the same qwen3.5 backbone; videojev =
    // qwen3.5 + a cosine decision head (score-only inference). vocab_target
    // 248320: the Qwen3.5 family is 248k - the shared 151k default made the
    // tokenizer discovery reject the checkpoint's own tokenizer.json whenever
    // a GGUF shipped without a readable vocab_size key (server garbage)
    { { "qwen3_5", "marlin", "videojev" }, create_qwen3_5,
      TokenizerMode::Qwen2Regex, "models/Qwen3.5-2B", 248320, false, false, "" },
    // MiniCPM-V 4.7 (openbmb, 1B dense / 35B-A3B MoE): the text trunk IS
    // qwen3_5_text, so it runs on the qwen3_5 backend's minicpm mode - SigLIP
    // canvas tower + 16x window/MLP mergers instead of the Qwen3-VL ViT.
    { { "minicpmv47", "minicpm_v4_7", nullptr }, create_qwen3_5,
      TokenizerMode::Qwen2Regex, "models/MiniCPM-V-4.7-1B", 248094, false, false, "" },
    // Qwen3-VL dense family (OneStreamer-4B = a Qwen3-VL-4B-Instruct SFT): its own
    // backend - 36 dense layers, no q|gate, IMROPE 24/20/20 over the full
    // head_dim, DeepStack (vision blocks 5/11/17 merged into text layers 0..2)
    { { "qwen3vl", "qwen3_vl", nullptr }, create_qwen3vl,
      TokenizerMode::Qwen2Regex, "models/OneStreamer-4B", 151936, false, false, "" },
    { { "moss_vl", nullptr }, create_moss_vl,
      TokenizerMode::Qwen2Regex, "models/Moss-VL-Realtime-q4k", 151936, false, false, "moss.html" },
    // SmolVLM (Idefics3): SigLIP tower + pixel-shuffle connector + SmolLM2 trunk
    { { "smolvlm", "idefics3", nullptr }, create_smolvlm,
      TokenizerMode::Gpt2Regex, "", 151936, false, false, "" },
    { { "qwen2vl", "qwen2_5_vl", nullptr }, create_qwen2vl,
      TokenizerMode::Qwen2Regex, "", 151936, false, false, "" },
    // video towers: labels or pooled embedding. ViViT is the same graph with a
    // cls token + learned position table, so it reuses the videomae backend.
    { { "videomae", "vivit", nullptr }, create_videomae,
      TokenizerMode::Qwen2Regex, "", 0, false, true, "" },
    // V-JEPA 2: the same shape of tower (pure video encoder -> embedding) but
    // with 3-D decomposed RoPE instead of a position table, so it gets its own
    // graph; no classifier release at fpc64-256, features only.
    { { "vjepa2", "vjepa", nullptr }, create_vjepa2,
      TokenizerMode::Qwen2Regex, "", 0, false, true, "" },
    // Mage-VL-4B: Mage-ViT codec ViT + Qwen3-4B trunk; realtime gate gguf
    // (mage-gate-f16.gguf) sits next to the model gguf and loads when present
    { { "mage_vl", nullptr }, create_mage_vl,
      TokenizerMode::Qwen2Regex, "models/Mage-VL", 151936, false, false, "" },
    // Qwen2.5-Omni thinker: Qwen2.5 trunk + Qwen2.5-VL tower + Whisper audio;
    // the talker gguf (speech output) is an optional sibling of the gguf
    { { "qwen25omni", "qwen2_5_omni", nullptr }, create_qwen25omni,
      TokenizerMode::Qwen2Regex, "models/Qwen2.5-Omni-3B", 151936, false, false, "omni.html" },
    // EmbeddingGemma-2: bidirectional text encoder (PLE) + Gemma4 vision/audio
    // towers fused as soft tokens; embedding output via Model::embed
    { { "embgemma2", nullptr }, create_embedgemma2,
      TokenizerMode::GemmaSP, "models/embeddinggemma-2", 262144, false, false, "" },
    // Gemma 4 12B (Gemma4Unified): mixed SWA/dense text trunk, text-only for
    // now - the vision/audio towers + soft-token fusion land in a later round
    { { "gemma4", nullptr }, create_gemma4,
      TokenizerMode::GemmaSP, "models/gemma-4-12B", 262144, false, false, "" },
    // SigLIP image tower: pooled embedding (get_image_features) / classifier
    { { "siglip", nullptr }, create_siglip,
      TokenizerMode::Qwen2Regex, "models/siglip-base", 0, false, true, "" },
    // Kwai Keye-VL-2.0-30B-A3B: Qwen3 MoE trunk + DSA sparse attention +
    // SigLIP tower; the mmproj gguf sits next to the main gguf (CPU by
    // default - the 19 GB Q4 checkpoint does not fit a 12 GB GPU)
    { { "keye-vl2", "keye_vl2", nullptr }, create_keye_vl,
      TokenizerMode::Qwen2Regex, "models/keye-vl2", 151936, false, false, "" },
    // Qwen3-Omni-30B-A3B thinker: Qwen3 full-attention trunk + routed MoE
    // (vision/audio towers staged; text-only for now)
    { { "qwen3omni", "qwen3_omni", nullptr }, create_qwen3omni,
      TokenizerMode::Qwen2Regex, "models/qwen3-omni", 152064, false, false, "" },
    // DeepSeek Janus-Pro (1B/7B, understanding path): timm SigLIP tower +
    // MLP aligner + llama trunk; the generation side (VQ tokenizer) is not
    // part of this engine
    { { "janus", "janus_pro", nullptr }, create_janus,
      TokenizerMode::DeepSeekBpe, "models/Janus-Pro-1B", 102400, false, false, "" },
    // NVIDIA Nemotron-3.5-Parse: C-RADIOv4-SO400M tower + conv neck + 10 layer
    // mBART prefix decoder. GPT-2 byte-level BPE base vocab (50k) plus 22.5k
    // added coordinate/class tokens, which the decode path must keep verbatim.
    { { "nemotron_parse", "nemotron-parse", nullptr }, create_nemotron_parse,
      TokenizerMode::Gpt2Regex, "models/Nemotron-3.5-Parse", 72512, false, false, "" },
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
    a.web_page = (e->web_page && e->web_page[0]) ? e->web_page : "vc3.html";
    return a;
}

// qwen3.5 aliases that ship as their own checkpoints get their own tokenizer
// dir instead of the shared default_dir above them in the arch table
std::string default_tokenizer_dir(const std::string & arch) {
    static const struct { const char * name, * dir; } kAliasDirs[] = {
        { "marlin",   "models/Marlin-2B" },
        { "videojev", "models/VideoJev-4B" },
    };
    for (const auto & a : kAliasDirs)
        if (arch == a.name) return a.dir;
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
