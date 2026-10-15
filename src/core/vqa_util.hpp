// vqa_util.hpp - small shared helpers for the VQA engine (no ggml dependency).
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace vqa {

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
