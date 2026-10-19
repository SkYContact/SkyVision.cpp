#include "core/text/utf8.hpp"

#include <mutex>

namespace vqa {

void append_utf8(std::string & out, uint32_t cp) {
    if (cp < 0x80) out += (char)cp;
    else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
    else { out += (char)(0xF0 | (cp >> 18)); out += (char)(0x80 | ((cp >> 12) & 0x3F)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
}

uint32_t utf8_next(const unsigned char * s, size_t len, size_t & i) {
    unsigned char c = s[i];
    if (c < 0x80) { i += 1; return c; }
    if ((c & 0xE0) == 0xC0) { uint32_t v = c & 0x1F; i += 1; if (i < len) { v = (v << 6) | (s[i] & 0x3F); i += 1; } return v; }
    if ((c & 0xF0) == 0xE0) { uint32_t v = c & 0x0F; i += 1; for (int k = 0; k < 2 && i < len; k++) { v = (v << 6) | (s[i] & 0x3F); i += 1; } return v; }
    uint32_t v = c & 0x07; i += 1;
    for (int k = 0; k < 3 && i < len; k++) { v = (v << 6) | (s[i] & 0x3F); i += 1; }
    return v;
}

std::vector<int> g_byte_to_cp;
std::vector<int> g_cp_to_byte;

void init_byte_table() {
    // multiple backends may load tokenizers concurrently on first use; the
    // once-flag closes the check/init race the old empty() guard had
    static std::once_flag once;
    std::call_once(once, [] {
        g_byte_to_cp.assign(256, 0);
        std::vector<int> cs;
        for (int b = 0; b < 256; b++) {
            if ((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF))
                { g_byte_to_cp[b] = b; cs.push_back(b); }
        }
        std::vector<bool> used(256, false);
        for (int b : cs) used[b] = true;
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!used[b]) { g_byte_to_cp[b] = 256 + n; cs.push_back(b); n++; }
        }
        g_cp_to_byte.assign(256 + n, -1);
        for (int b = 0; b < 256; b++) g_cp_to_byte[g_byte_to_cp[b]] = b;
    });
}

}  // namespace vqa
