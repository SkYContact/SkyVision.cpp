#include "tokenizer.hpp"

#include "unicode_ranges.inc"
#include "vqa_util.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace vqa {

// ---------------------------------------------------------------- unicode classes
static bool in_ranges(const uint32_t (*r)[2], int n, uint32_t cp) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (cp < r[mid][0]) hi = mid - 1;
        else if (cp > r[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}

static inline bool is_letter(uint32_t cp) { return in_ranges(vc3_uni_L, vc3_uni_L_N, cp); }
static inline bool is_number(uint32_t cp) { return in_ranges(vc3_uni_N, vc3_uni_N_N, cp); }
static inline bool is_space(uint32_t cp) { return in_ranges(vc3_uni_S, vc3_uni_S_N, cp); }

static inline uint32_t cp_at(const std::string & s, size_t & i) {
    return utf8_next((const unsigned char *)s.data(), s.size(), i);
}

static size_t scan_space(const std::string & s, size_t pos) {
    size_t k = pos;
    while (k < s.size()) {
        size_t k2 = k;
        if (!is_space(cp_at(s, k2))) break;
        k = k2;
    }
    return k;
}

static inline bool ascii_eq_ci(char a, char b) {
    if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
    if (b >= 'A' && b <= 'Z') b = (char)(b - 'A' + 'a');
    return a == b;
}

// ---------------------------------------------------------------- pre-tokenizer
// Byte length of the pre-token starting at text[pos], replicating the Rust
// `regex` crate's leftmost-first alternation for the Qwen2 pattern.
static size_t pretoken_len(const std::string & text, size_t pos) {
    const size_t n = text.size();
    const unsigned char * s = (const unsigned char *)text.data();

    size_t i = pos;
    uint32_t cp0 = utf8_next(s, n, i);
    size_t len0 = i - pos;

    // alt 1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (cp0 == '\'') {
        static const char * suffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
        for (const char * suf : suffixes) {
            size_t l = strlen(suf);
            if (pos + 1 + l <= n) {
                bool ok = true;
                for (size_t k = 0; k < l; k++)
                    if (!ascii_eq_ci(text[pos + 1 + k], suf[k])) { ok = false; break; }
                if (ok) return 1 + l;
            }
        }
    }

    // alt 2: [^\r\n\p{L}\p{N}]? \p{L}+
    {
        size_t j = pos;
        if (cp0 != '\r' && cp0 != '\n' && !is_letter(cp0) && !is_number(cp0)) j = i;
        size_t k = j, nletters = 0;
        while (k < n) {
            size_t k2 = k;
            if (!is_letter(cp_at(text, k2))) break;
            k = k2;
            nletters++;
        }
        if (nletters > 0) return k - pos;
    }

    // alt 3: \p{N}
    if (is_number(cp0)) return len0;

    // alt 4: ' '? [^\s\p{L}\p{N}]+ [\r\n]*
    {
        size_t j = pos;
        if (cp0 == ' ') j = i;
        size_t k = j, run = 0;
        while (k < n) {
            size_t k2 = k;
            uint32_t c2 = cp_at(text, k2);
            if (is_space(c2) || is_letter(c2) || is_number(c2)) break;
            k = k2;
            run++;
        }
        if (run > 0) {
            while (k < n && (text[k] == '\r' || text[k] == '\n')) k++;
            return k - pos;
        }
    }

    // whitespace alternatives. A UTF-8 continuation byte is always >= 0x80, so
    // scanning raw bytes for '\r'/'\n' inside a whitespace run is safe.
    if (is_space(cp0)) {
        const size_t k = scan_space(text, pos);
        // alt 5: \s* [\r\n]+
        size_t q = std::string::npos;
        for (size_t t = pos; t < k; t++) if (text[t] == '\r' || text[t] == '\n') q = t;
        if (q != std::string::npos) {
            size_t e = q;
            while (e < n && (text[e] == '\r' || text[e] == '\n')) e++;
            return e - pos;
        }
        // alt 6: \s+ (?!\S) -> when a non-space follows, backtrack one codepoint
        const size_t L = k - pos;
        if (k >= n) return L;
        return L >= 2 ? L - 1 : L;
    }

    return len0;   // every byte is covered by one of the alternatives
}

// ---------------------------------------------------------------- load
bool Tokenizer::load(const char * path, TokenizerMode mode) {
    std::string js;
    if (!read_file(path, js)) { fprintf(stderr, "tokenizer.json not found: %s\n", path); return false; }
    init_byte_table();

    use_pretokenizer = (mode == TokenizerMode::Qwen2Regex);
    use_added_tokens = (mode == TokenizerMode::Qwen2Regex);

    // vocab
    size_t vp = js.find("\"vocab\"");
    if (vp == std::string::npos) { fprintf(stderr, "no vocab in tokenizer.json\n"); return false; }
    vp = js.find('{', vp);
    int depth = 0;
    for (size_t i = vp; i < js.size(); i++) {
        char c = js[i];
        if (c == '"') {
            std::string key = json_parse_string(js, i);
            size_t j = js.find(':', i);
            if (j == std::string::npos) break;
            int id = atoi(js.c_str() + j + 1);
            if ((int)id_to_piece.size() <= id) id_to_piece.resize(id + 1);
            id_to_piece[id] = key;
            vocab[key] = id;
            i = j;
            continue;
        }
        if (c == '{' || c == '[') depth++;
        if (c == '}' || c == ']') { if (--depth == 0) break; }
    }

    // merges: [["A","B"], ...] (older exports use "A B" strings, handled too)
    size_t mp = js.find("\"merges\"");
    if (mp != std::string::npos) {
        mp = js.find('[', mp);
        bool have1 = false;
        std::string s1;
        int rank = 0, depth2 = 1;
        for (size_t i = mp + 1; i < js.size(); i++) {
            char c = js[i];
            if (c == '"') {
                std::string s = json_parse_string(js, i);
                i--;
                if (!have1) {
                    size_t sp = s.find(' ');
                    if (sp != std::string::npos) {          // "A B" form
                        merge_rank[{s.substr(0, sp), s.substr(sp + 1)}] = rank++;
                        have1 = false;
                    } else {
                        s1 = s;
                        have1 = true;
                    }
                } else {
                    merge_rank[{s1, s}] = rank++;
                    have1 = false;
                }
                continue;
            }
            if (c == '[') { depth2++; continue; }
            if (c == ']') { if (--depth2 == 0) break; continue; }
        }
    }

    // added tokens (specials)
    if (use_added_tokens) {
        size_t ap = js.find("\"added_tokens\"");
        if (ap != std::string::npos) {
            size_t arr = js.find('[', ap);
            int depth3 = 1;
            for (size_t i = arr + 1; i < js.size(); i++) {
                char c = js[i];
                if (c == '[' || c == '{') { depth3++; continue; }
                if (c == ']') { if (--depth3 == 0) break; continue; }
                if (c == '}') { depth3--; continue; }
                if (c == '"') {
                    std::string key = json_parse_string(js, i);
                    size_t colon = js.find(':', i);
                    if (colon == std::string::npos) break;
                    if (key == "id") {
                        int id = atoi(js.c_str() + colon + 1);
                        size_t cpos = js.find("\"content\"", colon);
                        if (cpos != std::string::npos) {
                            size_t q = js.find('"', js.find(':', cpos) + 1);
                            size_t save = q;
                            std::string content = json_parse_string(js, save);
                            added.push_back({content, id});
                            if ((int)id_to_piece.size() <= id) id_to_piece.resize(id + 1);
                            id_to_piece[id] = content;
                        }
                    }
                    i = colon;
                    continue;
                }
            }
        }
        std::sort(added.begin(), added.end(),
                  [](const std::pair<std::string, int> & a, const std::pair<std::string, int> & b) {
                      return a.first.size() > b.first.size();
                  });
    }
    is_added.assign(id_to_piece.size(), false);
    for (auto & a : added) if (a.second < (int)is_added.size()) is_added[a.second] = true;

    fprintf(stderr, "tokenizer: %zu tokens, %zu merges, %zu added (%s)\n",
            id_to_piece.size(), merge_rank.size(), added.size(),
            mode == TokenizerMode::Qwen2Regex ? "qwen2 regex" : "plain gpt2");
    return !id_to_piece.empty();
}

const std::string & Tokenizer::piece_of(int id) const {
    static const std::string empty;
    if (id < 0 || id >= (int)id_to_piece.size()) return empty;
    return id_to_piece[id];
}

int Tokenizer::id_of(const std::string & piece) const {
    auto it = vocab.find(piece);
    if (it != vocab.end()) return it->second;
    for (auto & a : added) if (a.first == piece) return a.second;
    return -1;
}

int Tokenizer::match_added(const std::string & text, size_t pos, size_t & len) const {
    for (auto & a : added) {
        if (a.first.empty()) continue;
        if (pos + a.first.size() <= text.size() &&
            memcmp(text.data() + pos, a.first.data(), a.first.size()) == 0) {
            len = a.first.size();
            return a.second;
        }
    }
    return -1;
}

// ---------------------------------------------------------------- encode
void Tokenizer::encode_piece(const std::string & piece, std::vector<int> & out) const {
    std::vector<std::string> parts;
    parts.reserve(piece.size());
    for (unsigned char c : piece) {
        std::string ch;
        append_utf8(ch, (uint32_t)g_byte_to_cp[c]);
        parts.push_back(ch);
    }
    while (parts.size() > 1) {
        int best_rank = 1 << 30;
        size_t best_i = 0;
        for (size_t i = 0; i + 1 < parts.size(); i++) {
            auto it = merge_rank.find({parts[i], parts[i + 1]});
            if (it != merge_rank.end() && it->second < best_rank) { best_rank = it->second; best_i = i; }
        }
        if (best_rank == (1 << 30)) break;
        parts[best_i] += parts[best_i + 1];
        parts.erase(parts.begin() + best_i + 1);
    }
    for (auto & p : parts) {
        auto it = vocab.find(p);
        if (it != vocab.end()) out.push_back(it->second);
    }
}

std::vector<int> Tokenizer::encode(const std::string & text) const {
    std::vector<int> out;
    size_t i = 0;
    while (i < text.size()) {
        if (use_added_tokens) {
            size_t alen = 0;
            int sid = match_added(text, i, alen);
            if (sid >= 0) { out.push_back(sid); i += alen; continue; }
        }
        size_t n = use_pretokenizer ? pretoken_len(text, i) : (text.size() - i);
        if (n == 0) { size_t j = i; cp_at(text, j); n = j - i; }
        encode_piece(text.substr(i, n), out);
        i += n;
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int> & ids) const {
    std::string out;
    for (int t : ids) {
        if (t < 0 || t >= (int)id_to_piece.size()) continue;
        if (t < (int)is_added.size() && is_added[t]) continue;   // specials are not text
        const std::string & piece = id_to_piece[t];
        size_t i = 0;
        while (i < piece.size()) {
            uint32_t cp = utf8_next((const unsigned char *)piece.data(), piece.size(), i);
            if (cp < g_cp_to_byte.size() && g_cp_to_byte[cp] >= 0) out += (char)g_cp_to_byte[cp];
            else append_utf8(out, cp);
        }
    }
    return out;
}

bool debug_tokenize(const char * tokenizer_json, const std::string & text,
                    std::vector<int> & ids, std::string & decoded, TokenizerMode mode) {
    Tokenizer t;
    if (!t.load(tokenizer_json, mode)) return false;
    ids = t.encode(text);
    decoded = t.decode(ids);
    return true;
}

}  // namespace vqa
