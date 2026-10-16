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

}  // namespace vqa

namespace vqa {

// The moondream3 engine still lives in its own binary (moondream-cli); it is wired
// behind this same interface in a follow-up step. Until then the factory reports it
// as unavailable instead of failing to link.
// create_moondream() is defined by the moondream backend adapter
// (moondream_vqa_adapter.inc) once linked into the same binary.

Model * vqa_create_model(const std::string & arch) {
    if (arch == "videochat3") return create_videochat3();
    if (arch == "qwen3_5" || arch == "marlin") return create_qwen3_5();   // marlin = legacy GGUF label
    if (arch == "moondream3") return create_moondream();
    return nullptr;
}

}  // namespace vqa
