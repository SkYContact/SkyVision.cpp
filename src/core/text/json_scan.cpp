#include "core/text/json_scan.hpp"

#include "core/text/utf8.hpp"

#include <cstdlib>

namespace vqa {

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

}  // namespace vqa
