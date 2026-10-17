// model_registry.hpp - model discovery + the single-resident model slot.
//
// Two front ends drive the engine (the Web UI and the MCP server) and both need
// the same three things: which GGUF files this build can actually run (known
// architecture *and* its side files present), a resolved tokenizer per model,
// and one loaded model instance at a time (a 9B MoE + its vision tower does not
// fit twice). That logic lives here instead of being copied into every front
// end - discovery carries platform-specific directory probing and a tokenizer
// vocab heuristic that must not drift between callers.
#pragma once

#include "vqa.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vqa {

// One runnable GGUF, with every runtime dependency already resolved.
struct ModelEntry {
    std::string id;            // file stem, the id clients address
    std::string path;          // gguf path
    std::string arch;          // general.architecture (normalised: marlin vs qwen3_5)
    std::string quant;         // general.file_type rendered as "fNN"
    uint64_t size = 0;         // bytes (0 when the filesystem would not say)
    std::string vision_path;   // moondream3: the shared *-vision.gguf sibling
    std::string tokenizer;     // resolved tokenizer.json ("" for pure video towers)
};

// Scan `dir` for *.gguf and keep the runnable ones. `tokenizer_override` (a
// file path) wins over the per-model probe; it is the Web UI's --tokenizer.
class ModelRegistry {
public:
    void discover(const std::string & dir, const std::string & tokenizer_override = std::string());

    // Register one explicit gguf path (outside the scanned directory).
    bool add_path(const std::string & path, std::string & err);

    const std::vector<ModelEntry> & entries() const { return entries_; }
    const std::string & dir() const { return dir_; }
    const ModelEntry * find(const std::string & id_or_path) const;

    const ModelEntry * find_loaded() const { return loaded_id_.empty() ? nullptr : find(loaded_id_); }
    Model * loaded() const { return loaded_; }
    const std::string & loaded_id() const { return loaded_id_; }

    // Load `e`, unloading whatever else is resident. `opt` carries caller
    // knobs (n_ctx / threads / cpu) plus an explicit vision override.
    Model * load(const ModelEntry & e, const LoadOptions & opt, std::string & err);
    void unload();

private:
    bool build_entry(const std::string & path, ModelEntry & e, std::string & why) const;
    // `vocab` is the model's own vocab size from the GGUF (0 when absent); the
    // candidate list is scored against it so the right tokenizer.json wins in a
    // directory that holds several checkpoints.
    std::string pick_tokenizer(const std::string & base_dir, const std::string & stem,
                               const std::string & arch, size_t vocab) const;

    std::vector<ModelEntry> entries_;
    std::string dir_ = ".";
    std::string tokenizer_override_;
    std::string vision_path_;   // shared tower for moondream weights in dir_
    Model * loaded_ = nullptr;
    std::string loaded_id_;
};

// Path existence without opening the file (wins over fopen for >10GB weights
// and for directories).
bool path_exists(const std::string & path);

}  // namespace vqa
