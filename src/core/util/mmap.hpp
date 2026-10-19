// mmap.hpp - read-only file mapping and whole-file reads.
//
// The two primitives every backend loads with: map the GGUF once (zero copy -
// the ggml tensors point straight into the mapping, see engine/weight_load.hpp)
// and slurp the small side files (tokenizer.json, config) into a std::string.
// Split out of the old vqa_util grab-bag so a backend that only needs a mapping
// does not drag in the tokenizer's byte tables.
#pragma once

#include <cstddef>
#include <string>

namespace vqa {

struct MMap {
    void * ptr = nullptr;
    size_t size = 0;
#ifdef _WIN32
    void * hfile = nullptr;
    void * hmap = nullptr;
#endif
    bool load(const char * path);
    bool map_file(const char * path);      // alias kept for readability
    void release();
    ~MMap() { release(); }

    // move-only: the destructor unmaps, so a shallow copy would leave two
    // owners and the first one to die unmaps the region out from under the
    // other (hit as a zero-copy weight segfault in mage_vl's gate loader).
    MMap() = default;
    MMap(const MMap &) = delete;
    MMap & operator=(const MMap &) = delete;
    MMap(MMap && o) noexcept { steal(o); }
    MMap & operator=(MMap && o) noexcept {
        if (this != &o) { release(); steal(o); }
        return *this;
    }
private:
    void steal(MMap & o) {
        ptr = o.ptr; size = o.size;
#ifdef _WIN32
        hfile = o.hfile; hmap = o.hmap;
        o.hfile = nullptr; o.hmap = nullptr;
#endif
        o.ptr = nullptr; o.size = 0;
    }
};

// whole-file read (returns false and leaves `out` empty on failure)
bool read_file(const char * path, std::string & out);

}  // namespace vqa
