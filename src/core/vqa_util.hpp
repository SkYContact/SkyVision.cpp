// vqa_util.hpp - small shared helpers for the VQA engine (no ggml dependency).
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace vqa {

// ---------------------------------------------------------------- aligned buffer
// Byte buffer whose data() is 64-byte aligned: ggml_backend_cpu_buffer_from_ptr
// requires TENSOR_ALIGNMENT (32) and std::vector's data() guarantees nothing
// beyond alignof(char) - glibc's mmap-backed large blocks land at page+16.
struct AlignedBuf {
    void resize(size_t n) {
        raw.resize(n + kAlign);
        ptr = reinterpret_cast<char *>((reinterpret_cast<uintptr_t>(raw.data()) + (kAlign - 1)) & ~uintptr_t(kAlign - 1));
        len = n;
    }
    bool empty() const { return len == 0; }
    size_t size() const { return len; }
    char * data() { return ptr; }
    const char * data() const { return ptr; }
    void clear() { raw.clear(); ptr = nullptr; len = 0; }
    void shrink_to_fit() { raw.shrink_to_fit(); }

private:
    static constexpr size_t kAlign = 64;
    std::vector<char> raw;
    char * ptr = nullptr;
    size_t len = 0;
};

// ---------------------------------------------------------------- mmap
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

// whole-file read (returns empty on failure)
bool read_file(const char * path, std::string & out);

// ---------------------------------------------------------------- json scan
// Parse a JSON string literal starting at js[i] == '"'; returns decoded content
// and advances i past the closing quote.
std::string json_parse_string(const std::string & js, size_t & i);

// Extract a top-level numeric / bool value for "key". Returns false if absent.
bool json_scan_number(const std::string & js, const char * key, double & out);

// ---------------------------------------------------------------- utf8
void append_utf8(std::string & out, uint32_t cp);
uint32_t utf8_next(const unsigned char * s, size_t len, size_t & i);

// ---------------------------------------------------------------- GPT-2 byte<->unicode
// vocab pieces are stored with the GPT-2 byte-level alphabet: byte -> codepoint.
void init_byte_table();
extern std::vector<int> g_byte_to_cp;    // 256 entries (byte -> codepoint)
extern std::vector<int> g_cp_to_byte;    // codepoint -> byte, -1 when unmapped

}  // namespace vqa
