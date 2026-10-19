// utf8.hpp - UTF-8 codec plus the GPT-2 byte<->unicode alphabet.
//
// Every supported tokenizer.json stores its vocab pieces in the GPT-2 byte level
// alphabet (each raw byte is written as a printable codepoint), so the encoder
// needs both directions: bytes -> text for decoding, text -> bytes for lookups.
// The tables are built once under call_once: several backends may load
// tokenizers concurrently on first use.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vqa {

// append one code point to a UTF-8 string
void append_utf8(std::string & out, uint32_t cp);

// decode one code point at s[i], advancing i (returns 0 at the end)
uint32_t utf8_next(const unsigned char * s, size_t len, size_t & i);

// build the GPT-2 byte<->codepoint tables (idempotent)
void init_byte_table();
extern std::vector<int> g_byte_to_cp;    // 256 entries (byte -> codepoint)
extern std::vector<int> g_cp_to_byte;    // codepoint -> byte, -1 when unmapped

}  // namespace vqa
