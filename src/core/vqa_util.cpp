#include "vqa_util.hpp"

#include <cstdlib>
#include <sys/stat.h>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#else
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <unistd.h>
#endif

namespace vqa {

bool MMap::load(const char * path) {
#ifdef _WIN32
    HANDLE hf = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hf == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER li;
    if (!GetFileSizeEx(hf, &li) || li.QuadPart <= 0) { CloseHandle(hf); return false; }
    HANDLE hm = CreateFileMappingA(hf, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hm) { CloseHandle(hf); return false; }
    void * p = MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
    if (!p) { CloseHandle(hm); CloseHandle(hf); return false; }
    ptr = p;
    size = (size_t)li.QuadPart;
    hfile = hf;
    hmap = hm;
    return true;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); return false; }
    void * p = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return false;
    ptr = p;
    size = (size_t)st.st_size;
    return true;
#endif
}

bool MMap::map_file(const char * path) { return load(path); }

void MMap::release() {
#ifdef _WIN32
    if (ptr) { UnmapViewOfFile(ptr); ptr = nullptr; }
    if (hmap) { CloseHandle((HANDLE)hmap); hmap = nullptr; }
    if (hfile) { CloseHandle((HANDLE)hfile); hfile = nullptr; }
#else
    if (ptr) { munmap(ptr, size); ptr = nullptr; }
#endif
    size = 0;
}

bool read_file(const char * path, std::string & out) {
    FILE * f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return false; }
    out.resize((size_t)sz);
    size_t got = fread(out.data(), 1, (size_t)sz, f);
    fclose(f);
    if (got != (size_t)sz) { out.clear(); return false; }
    return true;
}

std::string json_parse_string(const std::string & js, size_t & i) {
    std::string out;
    i++;    // skip opening quote
    for (; i < js.size(); i++) {
        char c = js[i];
        if (c == '"') { i++; return out; }
        if (c != '\\') { out += c; continue; }
        char e = js[++i];
        switch (e) {
            case 'n': out += '\n'; break;
            case 't': out += '\t'; break;
            case 'r': out += '\r'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'u': {
                if (i + 4 >= js.size()) return out;
                uint32_t cp = (uint32_t)strtol(js.substr(i + 1, 4).c_str(), nullptr, 16);
                i += 4;
                if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 < js.size() && js[i + 1] == 'u') {
                    uint32_t lo = (uint32_t)strtol(js.substr(i + 3, 4).c_str(), nullptr, 16);
                    if (lo >= 0xDC00 && lo <= 0xDFFF) { cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); i += 6; }
                }
                append_utf8(out, cp);
                break;
            }
            default: out += e;
        }
    }
    return out;
}

bool json_scan_number(const std::string & js, const char * key, double & out) {
    std::string pat = std::string("\"") + key + "\"";
    size_t p = js.find(pat);
    if (p == std::string::npos) return false;
    size_t c = js.find(':', p + pat.size());
    if (c == std::string::npos) return false;
    const char * s = js.c_str() + c + 1;
    while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') s++;
    if (!(*s == '-' || (*s >= '0' && *s <= '9'))) return false;
    out = atof(s);
    return true;
}

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
    if (!g_byte_to_cp.empty()) return;
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
}

}  // namespace vqa
