// load_util.hpp - the load() preamble every backend had copy-pasted:
// LoadOptions -> members, the per-backend <PREFIX>_DUMP_DIR dev override, and
// resolving chat-template token ids against the loaded vocabulary.
//
// The env name stays backend-specific on purpose (`VC3_DUMP_DIR`,
// `QWEN35_DUMP_DIR`, `MOSS_DUMP_DIR`, `VIDEOMAE_DUMP_DIR`, ...) - scripts and
// READMEs already reference them, so only the plumbing is shared.
#pragma once

#include "tokenizer.hpp"
#include "vqa.hpp"

#include <cstdlib>
#include <initializer_list>
#include <string>
#include <utility>

namespace vqa {

// Context window: keep the backend's baked-in default unless the caller asked
// for a specific size.
inline int resolve_n_ctx(const LoadOptions & opt, int def) {
    return opt.n_ctx > 0 ? opt.n_ctx : def;
}

// n_threads / use_gpu / dump_dir from LoadOptions, then `<prefix>_DUMP_DIR`
// (when set) overrides the dump directory.
inline void apply_load_options(const LoadOptions & opt, const char * env_prefix,
                              int & n_threads, bool & use_gpu, std::string & dump_dir) {
    n_threads = opt.n_threads;
    use_gpu = opt.use_gpu;
    dump_dir = opt.dump_dir;
    if (env_prefix && *env_prefix) {
        const std::string key = std::string(env_prefix) + "_DUMP_DIR";
        if (const char * d = getenv(key.c_str())) dump_dir = d;
    }
}

// Token id for a piece, or `def` when the vocabulary does not carry it (the
// converters bake in the defaults, so a missing special token is not fatal).
inline int pick_token(const Tokenizer & tok, const char * piece, int def) {
    const int v = tok.id_of(piece);
    return v >= 0 ? v : def;
}

// Resolve a batch of "piece string -> member id" pairs in one go.
inline void assign_token_ids(const Tokenizer & tok,
                             std::initializer_list<std::pair<const char *, int *>> ids) {
    for (const auto & p : ids) *p.second = pick_token(tok, p.first, *p.second);
}

}  // namespace vqa
