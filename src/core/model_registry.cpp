// model_registry.cpp - model discovery + the single-resident model slot.
#include "model_registry.hpp"

#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
#else
  #include <dirent.h>
  #include <sys/stat.h>
#endif

namespace vqa {
namespace {

struct DirItem {
    std::string name;
    bool is_dir = false;
    uint64_t size = 0;
};

// one directory listing, sorted by name so ids/order are stable across runs
std::vector<DirItem> list_dir(const std::string & dir) {
    std::vector<DirItem> out;
#ifdef _WIN32
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA((dir + "/*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return out;
    do {
        DirItem it;
        it.name = fd.cFileName;
        it.is_dir = (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
        it.size = ((uint64_t)fd.nFileSizeHigh << 32) | (uint64_t)fd.nFileSizeLow;
        out.push_back(std::move(it));
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    DIR * d = opendir(dir.c_str());
    if (!d) return out;
    while (struct dirent * de = readdir(d)) {
        const std::string name = de->d_name;
        if (name == "." || name == "..") continue;
        DirItem it;
        it.name = name;
        struct stat st {};
        if (stat((dir + "/" + name).c_str(), &st) == 0) {
            it.is_dir = S_ISDIR(st.st_mode);
            it.size = (uint64_t)st.st_size;
        }
        out.push_back(std::move(it));
    }
    closedir(d);
#endif
    std::sort(out.begin(), out.end(), [](const DirItem & a, const DirItem & b) { return a.name < b.name; });
    return out;
}

bool ends_with(const std::string & s, const char * suffix) {
    const size_t n = strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

std::string stem_of(const std::string & name) {
    const size_t dot = name.rfind('.');
    return dot == std::string::npos ? name : name.substr(0, dot);
}

// "models/moondream3.1-q6k.gguf" -> "models", "x.gguf" -> "."
std::string dirname_of(const std::string & path) {
    const size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) return ".";
    if (slash == 0) return path.substr(0, 1);
    return path.substr(0, slash);
}

uint64_t file_size(const std::string & path) {
#ifdef _WIN32
    WIN32_FILE_ATTRIBUTE_DATA fa;
    if (!GetFileAttributesExA(path.c_str(), GetFileExInfoStandard, &fa)) return 0;
    return ((uint64_t)fa.nFileSizeHigh << 32) | (uint64_t)fa.nFileSizeLow;
#else
    struct stat st {};
    if (stat(path.c_str(), &st) != 0) return 0;
    return (uint64_t)st.st_size;
#endif
}

// general.architecture (+ general.file_type as "fNN"), with the legacy
// qwen3.5/marlin label swap resolved: marlin.skills is authoritative for which
// fine-tune a checkpoint is, because old converters labelled both "marlin".
// `vocab` is "<arch>.vocab_size" (0 when the file does not carry it): discovery
// matches a tokenizer.json against it instead of guessing by arch family.
bool gguf_arch_of(const std::string & path, std::string & arch, std::string & ftype, size_t & vocab) {
    struct gguf_init_params p = { true, nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), p);
    if (!g) return false;
    const int k = gguf_find_key(g, "general.architecture");
    if (k >= 0) arch = gguf_get_val_str(g, k);
    const std::string raw_arch = arch;
    const int ks = gguf_find_key(g, "marlin.skills");
    if (arch == "qwen3_5" || arch == "marlin")
        arch = (ks >= 0 && gguf_get_val_bool(g, ks)) ? "marlin" : "qwen3_5";
    // the two labels swap across converter generations (938a21c): the metadata
    // keys may carry either name regardless of general.architecture, so probe
    // both before giving up - a missed vocab_size makes discovery match a
    // 151k-vocab tokenizer against a 248k-vocab model (garbled output)
    int kv = gguf_find_key(g, (arch + ".vocab_size").c_str());
    if (kv < 0) kv = gguf_find_key(g, (raw_arch + ".vocab_size").c_str());
    if (kv < 0 && (arch == "qwen3_5" || arch == "marlin")) {
        // arch and raw_arch coincide when the file is already labelled
        // qwen3_5, yet the converter still wrote the legacy marlin.* keys
        // (models/Qwen3.5-2B ships marlin.vocab_size=248320) - probe the
        // sibling spelling too or the lookup falls to the family fallback,
        // whose 151k default then rejects the checkpoint's own tokenizer
        kv = gguf_find_key(g, arch == "qwen3_5" ? "marlin.vocab_size"
                                                : "qwen3_5.vocab_size");
    }
    if (kv >= 0) vocab = (size_t)gguf_get_val_u32(g, kv);
    const int kf = gguf_find_key(g, "general.file_type");
    if (kf >= 0) {
        char b[32];
        snprintf(b, sizeof(b), "f%d", (int)gguf_get_val_u32(g, kf));
        ftype = b;
    }
    gguf_free(g);
    return !arch.empty();
}

// Vocab size of a tokenizer.json, approximated by counting "key": entries
// between "vocab" and "merges" (the file is one flat object per section, so the
// count tracks the real vocab closely enough to tell the families apart).
size_t tokenizer_vocab(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return 0;
    std::string js;
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) js.append(buf, n);
    fclose(f);
    const size_t vp = js.find("\"vocab\"");
    if (vp == std::string::npos) return 0;
    size_t me = js.find("\"merges\"", vp);
    if (me == std::string::npos) me = js.size();
    return (size_t)std::count(js.begin() + (long)vp, js.begin() + (long)me, ':');
}

}  // namespace

bool path_exists(const std::string & path) {
#ifdef _WIN32
    return GetFileAttributesA(path.c_str()) != INVALID_FILE_ATTRIBUTES;
#else
    struct stat st {};
    return stat(path.c_str(), &st) == 0;
#endif
}

std::string ModelRegistry::pick_tokenizer(const std::string & base_dir, const std::string & stem,
                                          const std::string & arch, size_t vocab) const {
    if (!tokenizer_override_.empty() && path_exists(tokenizer_override_)) return tokenizer_override_;
    // candidate directories: the model's own directory first, then the scanned
    // models dir (they differ when a single path was registered explicitly)
    std::vector<std::string> dirs = { base_dir };
    if (dir_ != base_dir && !dir_.empty()) dirs.push_back(dir_);
    // last resort: the arch's canonical checkpoint dir (models/VideoChat3-4B
    // etc.) - covers single ggufs parked anywhere else in the repo
    std::string hint_dir = default_tokenizer_dir(arch);
    if (!hint_dir.empty())
        for (const auto & d : dirs) if (d != hint_dir) { dirs.push_back(hint_dir); break; }
    std::vector<std::string> cands;
    for (const auto & d : dirs) {
        cands.push_back(d + "/" + stem + "/tokenizer.json");   // per-model checkpoint dir
        cands.push_back(d + "/tokenizer.json");
        for (const auto & it : list_dir(d))
            if (it.is_dir) cands.push_back(d + "/" + it.name + "/tokenizer.json");
    }
    // A directory of checkpoints holds several tokenizers, and taking the first
    // one is wrong (a 150k Qwen vocab parses fine for moondream but decodes
    // garbage). Match against the model's own vocab size from the GGUF, which
    // every converter in this repo writes as "<arch>.vocab_size".
    const size_t target = vocab ? vocab : fallback_vocab_target(arch);
    // a tokenizer shipped inside the checkpoint's own directory belongs to that
    // checkpoint by definition: same-size siblings can still differ in their
    // added tokens (moss_vl's <|image_pad|>/<|silence|> exist only in its own)
    const std::string own_suffix = "/" + stem + "/tokenizer.json";
    for (const auto & c : cands) {
        if (!ends_with(c, own_suffix.c_str())) continue;
        const size_t n = tokenizer_vocab(c);
        if (n == 0) continue;
        const size_t diff = n > target ? n - target : target - n;
        if (diff * 5 <= target) return c;
    }
    // the arch's canonical checkpoint dir beats a same-size stranger: total
    // vocab counts collide across families (moss_vl and qwen25omni both parse
    // as 151673) while their added-token sets differ - a <|IMAGE|> missing
    // from the picked tokenizer silently splits the soft-token placeholders
    // and starves the injection indices (garbage answers + CUDA crash)
    if (!hint_dir.empty()) {
        const std::string c = hint_dir + "/tokenizer.json";
        const size_t n = tokenizer_vocab(c);
        if (n != 0) {
            const size_t diff = n > target ? n - target : target - n;
            if (diff * 5 <= target) return c;
        }
    }
    size_t best_diff = 0;
    std::string best;
    for (const auto & c : cands) {
        const size_t n = tokenizer_vocab(c);
        if (n == 0) continue;
        // a tokenizer within 20% of the model's vocab is the right one; anything
        // further off belongs to a different checkpoint family
        const size_t diff = n > target ? n - target : target - n;
        if (diff * 5 > target) continue;
        if (best.empty() || diff < best_diff) {
            best = c;
            best_diff = diff;
        }
    }
    return best;
}

bool ModelRegistry::build_entry(const std::string & path, ModelEntry & e, std::string & why) const {
    std::string arch, ftype;
    size_t vocab = 0;
    if (!gguf_arch_of(path, arch, ftype, vocab)) { why = "unreadable GGUF (or not a GGUF)"; return false; }
    const ArchInfo info = arch_info(arch);   // the backend list lives in vqa_registry
    if (!info.supported) {
        why = "architecture '" + arch + "' is not wired into this build (or is a side file)";
        return false;
    }
    const std::string base = dirname_of(path);
    const std::string file = path.substr(path.find_last_of("/\\") + 1);
    const std::string stem = stem_of(file);

    std::string vision, tokenizer;
    if (info.needs_vision_gguf) {
        // the vision tower is shared by every quantisation of the same release,
        // so it is looked up next to the weights, not per file name
        if (base == dir_ && !vision_path_.empty()) {
            vision = vision_path_;
        } else {
            for (const auto & it : list_dir(base))
                if (!it.is_dir && ends_with(it.name, "-vision.gguf")) { vision = base + "/" + it.name; break; }
        }
        if (vision.empty()) { why = "needs a *-vision.gguf next to it"; return false; }
    }
    if (info.needs_tokenizer) {
        tokenizer = pick_tokenizer(base, stem, arch, vocab);
        if (tokenizer.empty()) { why = "needs tokenizer.json"; return false; }
    }

    e.id = stem;
    e.path = path;
    e.arch = arch;
    e.quant = ftype;
    e.size = file_size(path);
    e.vision_path = vision;
    e.tokenizer = tokenizer;
    return true;
}

void ModelRegistry::discover(const std::string & dir, const std::string & tokenizer_override) {
    dir_ = dir.empty() ? std::string(".") : dir;
    while (dir_.size() > 1 && (dir_.back() == '/' || dir_.back() == '\\')) dir_.pop_back();
    tokenizer_override_ = tokenizer_override;
    entries_.clear();
    vision_path_.clear();

    std::vector<std::string> ggufs;
    for (const auto & it : list_dir(dir_)) {
        if (it.is_dir || !ends_with(it.name, ".gguf")) continue;
        ggufs.push_back(it.name);
        if (vision_path_.empty() && ends_with(it.name, "-vision.gguf")) vision_path_ = dir_ + "/" + it.name;
    }
    // HF-style layout: models/<checkpoint>/<model>.gguf - scan one level of
    // subdirectories too (pick_tokenizer already prefers a tokenizer.json
    // sitting in the gguf's own directory)
    for (const auto & it : list_dir(dir_)) {
        if (!it.is_dir || it.name[0] == '.') continue;
        const std::string sub = dir_ + "/" + it.name;
        for (const auto & g : list_dir(sub)) {
            if (g.is_dir || !ends_with(g.name, ".gguf")) continue;
            ggufs.push_back(it.name + "/" + g.name);
            if (vision_path_.empty() && ends_with(g.name, "-vision.gguf"))
                vision_path_ = sub + "/" + g.name;
        }
    }
    for (const auto & g : ggufs) {
        ModelEntry e;
        std::string why;
        if (!build_entry(dir_ + "/" + g, e, why)) {
            fprintf(stderr, "models: skip %s (%s)\n", g.c_str(), why.c_str());
            continue;
        }
        entries_.push_back(std::move(e));
    }
    fprintf(stderr, "models: %zu runnable model(s) in %s\n", entries_.size(), dir_.c_str());
}

bool ModelRegistry::add_path(const std::string & path, std::string & err) {
    if (!path_exists(path)) { err = "no such file: " + path; return false; }
    ModelEntry e;
    std::string why;
    if (!build_entry(path, e, why)) { err = why; return false; }
    for (auto & old : entries_)
        if (old.id == e.id) { old = e; return true; }
    entries_.push_back(std::move(e));
    return true;
}

const ModelEntry * ModelRegistry::find(const std::string & id_or_path) const {
    if (id_or_path.empty()) return nullptr;
    for (const auto & e : entries_)
        if (e.id == id_or_path) return &e;
    for (const auto & e : entries_)
        if (e.path == id_or_path) return &e;
    return nullptr;
}

Model * ModelRegistry::load(const ModelEntry & e, const LoadOptions & opt, std::string & err) {
    if (loaded_ && loaded_id_ == e.id) return loaded_;
    // VRAM guard (roadmap §6.3-E): predict the requirement before touching the
    // backend, so an over-budget load fails here with advice instead of as an
    // allocation failure three stacks deep (Windows: CUDA error 1455).
    std::string why;
    if (!vram_admits(vram_check(e, opt), why)) { err = why; return nullptr; }
    unload();
    fprintf(stderr, "registry: loading %s (%s) ...\n", e.id.c_str(), e.arch.c_str());
    fflush(stderr);
    Model * m = vqa_create_model(e.arch);
    if (!m) { err = "no backend wired for architecture '" + e.arch + "'"; return nullptr; }
    LoadOptions o = opt;
    if (o.vision_path.empty()) o.vision_path = e.vision_path;
    if (!m->load(e.path, e.tokenizer, o)) {
        err = "model load failed: " + e.path;
        delete m;
        return nullptr;
    }
    loaded_ = m;
    loaded_id_ = e.id;
    fprintf(stderr, "registry: %s ready\n", e.id.c_str());
    fflush(stderr);
    return m;
}

void ModelRegistry::unload() {
    if (!loaded_) return;
    fprintf(stderr, "registry: unloading %s\n", loaded_id_.c_str());
    delete loaded_;   // frees the backend buffers (one model resident at a time)
    loaded_ = nullptr;
    loaded_id_.clear();
}

}  // namespace vqa
