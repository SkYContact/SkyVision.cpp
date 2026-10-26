// np_markup.hpp - the Nemotron-3.5-Parse answer format, parsed and rendered the
// way the upstream package does it (nemotron_parse/engine/markdown.py,
// engine/policies.py and nemotron_parse/postprocessing.py).
//
// The model answers with a control-token stream, one element per block:
//
//   <x_0.0586><y_0.0539>Quarterly Report 2026<x_0.1621><y_0.0633><class_Text>
//   \__ opening (x1,y1) _/\_____ body _____/\__ closing (x2,y2) _/<class_..>
//
// Coordinates are content-normalized (padding already cropped), so pixels are
// just `value * original_size`. Block order IS the reading order - nothing here
// re-sorts by geometry.
//
// Deliberate simplifications against the upstream package (all presentation-only,
// none change the block/bbox contract):
//   * no chart panel rendering / caption relocation / native subtitles
//   * header promotion uses the shape heuristic only (upstream also wants the
//     first row to be mostly bold, which LaTeX tabular markup rarely carries)
//   * `raw_out` is not echoed back inside the JSON
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace nemotron_parse {

struct Block {
    std::string cls;
    double x1 = 0.0, y1 = 0.0, x2 = 0.0, y2 = 0.0;
    std::string text;   // literal body, untrimmed (upstream keeps it verbatim too)
};

namespace markup_detail {

inline bool is_num_start(const std::string & s, size_t p) {
    return p < s.size() && s[p] >= '0' && s[p] <= '9';
}

// `<tag_123.45>` -> value, next = index just past '>'
inline bool read_coord(const std::string & s, size_t pos, const char * tag, double & out,
                       size_t & next) {
    const size_t n = strlen(tag);
    if (s.compare(pos, n, tag) != 0) return false;
    size_t p = pos + n;
    const size_t start = p;
    while (p < s.size() && s[p] != '>') {
        const char ch = s[p];
        const bool ok = (ch >= '0' && ch <= '9') || ch == '.';
        if (!ok) return false;
        p++;
    }
    if (p >= s.size() || p == start) return false;
    const std::string num = s.substr(start, p - start);
    if (num.find('.') != num.rfind('.')) return false;   // one dot at most
    out = strtod(num.c_str(), nullptr);
    next = p + 1;
    return true;
}

// closing tail: `<x_a><y_b><class_Name>`
inline bool read_close(const std::string & s, size_t pos, double & x2, double & y2,
                       std::string & cls, size_t & next) {
    size_t p = 0;
    if (!read_coord(s, pos, "<x_", x2, p)) return false;
    if (!read_coord(s, p, "<y_", y2, p)) return false;
    static const char kCls[] = "<class_";
    const size_t n = sizeof(kCls) - 1;
    if (s.compare(p, n, kCls) != 0) return false;
    p += n;
    const size_t start = p;
    while (p < s.size() && s[p] != '>') p++;
    if (p >= s.size() || p == start) return false;
    cls = s.substr(start, p - start);
    next = p + 1;
    return true;
}

inline std::string trim(const std::string & s) {
    size_t a = 0, b = s.size();
    while (a < b && (unsigned char)s[a] <= ' ') a++;
    while (b > a && (unsigned char)s[b - 1] <= ' ') b--;
    return s.substr(a, b - a);
}

inline bool starts_with(const std::string & s, const char * p) {
    return s.compare(0, strlen(p), p) == 0;
}

inline std::string replace_all(std::string s, const std::string & from, const std::string & to) {
    if (from.empty()) return s;
    size_t p = 0;
    while ((p = s.find(from, p)) != std::string::npos) {
        s.replace(p, from.size(), to);
        p += to.size();
    }
    return s;
}

inline std::string html_escape(const std::string & s) {
    std::string o;
    o.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '&': o += "&amp;"; break;
            case '<': o += "&lt;"; break;
            case '>': o += "&gt;"; break;
            default: o += c;
        }
    }
    return o;
}

inline std::string json_escape(const std::string & s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (size_t i = 0; i < s.size(); i++) {
        const unsigned char c = (unsigned char)s[i];
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) {
                    char b[8];
                    snprintf(b, sizeof(b), "\\u%04x", c);
                    o += b;
                } else {
                    o += (char)c;
                }
        }
    }
    return o;
}

// strip a `\cmd{...}` wrapper, returning the payload and the index past its `}`
inline bool take_braced(const std::string & s, size_t open, std::string & body, size_t & next) {
    if (open >= s.size() || s[open] != '{') return false;
    int depth = 0;
    for (size_t p = open; p < s.size(); p++) {
        if (s[p] == '{') depth++;
        else if (s[p] == '}') {
            if (--depth == 0) {
                body = s.substr(open + 1, p - open - 1);
                next = p + 1;
                return true;
            }
        }
    }
    return false;
}

inline std::string inline_tex(const std::string & in) {
    std::string s = in;
    // inline emphasis first (payload is escaped afterwards by the caller)
    const struct { const char * cmd; const char * tag; } kMap[] = {
        { "\\textbf{", "b" }, { "\\textit{", "i" }, { "\\emph{", "i" },
        { "\\underline{", "u" }, { "\\texttt{", "code" },
    };
    for (const auto & m : kMap) {
        const std::string cmd = m.cmd;
        size_t p = 0;
        while ((p = s.find(cmd, p)) != std::string::npos) {
            std::string body;
            size_t next = 0;
            if (!take_braced(s, p + cmd.size() - 1, body, next)) { p += cmd.size(); continue; }
            const std::string open = std::string("<") + m.tag + ">";
            const std::string close = std::string("</") + m.tag + ">";
            s = s.substr(0, p) + open + body + close + s.substr(next);
            p += open.size() + body.size() + close.size();
        }
    }
    // a stray backslash escape (`\%`, `\&`, `\_`, `\$`) keeps the character
    std::string o;
    o.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '\\' && i + 1 < s.size() &&
            strchr("%&_$#{}", s[i + 1]) != nullptr) {
            o += s[i + 1];
            i++;
            continue;
        }
        o += s[i];
    }
    return o;
}

struct TableRow {
    std::vector<std::string> cells;      // already HTML (may contain <b> ...)
    std::vector<int> span;               // colspan per cell (>=1)
    std::vector<int> rowspan;            // rowspan per cell (>=1)
};

inline void split_cells(const std::string & row, TableRow & out) {
    size_t p = 0, start = 0;
    int brace_depth = 0;
    auto flush = [&](size_t end) {
        std::string cell = trim(row.substr(start, end - start));
        int colspan = 1, rowspan = 1;
        // \multicolumn{n}{spec}{text} / \multirow{n}{width}{text}
        for (;;) {
            std::string num, spec, body;
            size_t n1 = 0, n2 = 0, n3 = 0;
            if (starts_with(cell, "\\multicolumn")) {
                const size_t o = cell.find('{');
                if (!take_braced(cell, o, num, n1)) break;
                const size_t o2 = cell.find('{', n1);
                if (!take_braced(cell, o2, spec, n2)) break;
                const size_t o3 = cell.find('{', n2);
                if (!take_braced(cell, o3, body, n3)) break;
                colspan = std::max(1, atoi(num.c_str()));
                cell = trim(cell.substr(n3));
                continue;
            }
            if (starts_with(cell, "\\multirow")) {
                const size_t o = cell.find('{');
                if (!take_braced(cell, o, num, n1)) break;
                const size_t o2 = cell.find('{', n1);
                if (!take_braced(cell, o2, spec, n2)) break;
                const size_t o3 = cell.find('{', n2);
                if (!take_braced(cell, o3, body, n3)) break;
                rowspan = std::max(1, atoi(num.c_str()));
                cell = trim(cell.substr(n3));
                continue;
            }
            break;
        }
        out.cells.push_back(html_escape(inline_tex(cell)));
        out.span.push_back(colspan);
        out.rowspan.push_back(rowspan);
    };
    for (; p < row.size(); p++) {
        const char c = row[p];
        if (c == '{') brace_depth++;
        else if (c == '}') brace_depth = std::max(0, brace_depth - 1);
        else if (c == '&' && brace_depth == 0) {
            flush(p);
            start = p + 1;
        }
    }
    flush(row.size());
}

// `\begin{tabular}{...} ... \end{tabular}` -> HTML table. Returns false when the
// markup does not look like a table this simple converter understands (the
// caller then keeps the LaTeX verbatim, which is what upstream does too).
inline bool latex_table_to_html(const std::string & src, std::string & html) {
    const std::string kBegin = "\\begin{tabular}";
    const std::string kEnd = "\\end{tabular}";
    const size_t b = src.find(kBegin);
    if (b == std::string::npos) return false;
    const size_t e = src.find(kEnd, b + kBegin.size());
    if (e == std::string::npos) return false;
    size_t p = b + kBegin.size();
    // optional [pos], then the column spec
    if (p < src.size() && src[p] == '[') {
        const size_t close = src.find(']', p);
        if (close == std::string::npos) return false;
        p = close + 1;
    }
    while (p < src.size() && (src[p] == ' ' || src[p] == '\n' || src[p] == '\t')) p++;
    std::string spec;
    size_t after = 0;
    if (!take_braced(src, p, spec, after)) return false;
    std::string body = src.substr(after, e - after);

    // drop rules / spacing commands
    static const char * const kDrop[] = { "\\hline", "\\toprule", "\\midrule",
                                          "\\bottomrule", "\\arraybackslash", nullptr };
    for (int i = 0; kDrop[i]; i++) body = replace_all(body, kDrop[i], "");
    for (;;) {
        const size_t c = body.find("\\cline{");
        if (c == std::string::npos) break;
        const size_t close = body.find('}', c);
        if (close == std::string::npos) break;
        body.erase(c, close - c + 1);
    }

    // rows split on `\\` (the literal two-character row separator)
    std::vector<std::string> raw_rows;
    {
        size_t start = 0;
        for (size_t i = 0; i + 1 < body.size(); i++) {
            if (body[i] == '\\' && body[i + 1] == '\\') {
                raw_rows.push_back(body.substr(start, i - start));
                i++;
                start = i + 1;
            }
        }
        raw_rows.push_back(body.substr(start));
    }

    std::vector<TableRow> rows;
    int width = 0;
    for (const auto & rr : raw_rows) {
        if (trim(rr).empty()) continue;
        TableRow tr;
        split_cells(rr, tr);
        if (tr.cells.empty()) continue;
        int w = 0;
        for (size_t i = 0; i < tr.cells.size(); i++) w += tr.span[i];
        if (width == 0) width = w;
        if (w != width) return false;      // ragged -> bail, keep the LaTeX
        rows.push_back(std::move(tr));
    }
    if (rows.empty()) return false;

    // upstream promotes a header row when the shape says "header + data"; the
    // bold half of that test is skipped here (LaTeX tabular rarely marks bold)
    bool header = rows.size() >= 2 && rows[0].cells.size() >= 2 &&
                  rows[0].cells.size() <= 20 && rows[1].cells.size() >= 2;
    if (header) {
        int textual = 0;
        for (const auto & c : rows[0].cells) {
            const std::string t = trim(c);
            if (t.size() > 100) { header = false; break; }
            bool numeric = !t.empty();
            for (char ch : t) {
                const bool ok = (ch >= '0' && ch <= '9') || ch == '.' || ch == ',' ||
                                ch == '%' || ch == '$' || ch == ' ' || ch == '-' ||
                                ch == '+' || ch == '(' || ch == ')';
                if (!ok) { numeric = false; break; }
            }
            if (!t.empty() && !numeric) textual++;
        }
        if (header && textual * 10 < (int)rows[0].cells.size() * 6) header = false;
    }

    std::string o = "<table>";
    for (size_t r = 0; r < rows.size(); r++) {
        const bool th = header && r == 0;
        o += th ? "<thead><tr>" : (r == (header ? 1u : 0u) ? "<tbody><tr>" : "<tr>");
        for (size_t c = 0; c < rows[r].cells.size(); c++) {
            const char * tag = th ? "th" : "td";
            o += std::string("<") + tag;
            if (rows[r].span[c] > 1) o += " colspan=\"" + std::to_string(rows[r].span[c]) + "\"";
            if (rows[r].rowspan[c] > 1) o += " rowspan=\"" + std::to_string(rows[r].rowspan[c]) + "\"";
            o += std::string(">") + rows[r].cells[c] + "</" + tag + ">";
        }
        o += "</tr>";
        if (th) o += "</thead>";
    }
    if (!header) o += "</tbody>";
    o += "</table>";
    html = o;
    return true;
}

}  // namespace markup_detail

// The model occasionally drops the leading `<x_..><y_..>` of its very first
// element; upstream repairs that before parsing (repair_empty_prefix).
inline std::string repair_empty_prefix(const std::string & raw) {
    if (raw.empty()) return raw;
    const size_t x = raw.find("<x_");
    if (x == std::string::npos) return raw;
    if (x == 0) return raw;
    if (!markup_detail::trim(raw.substr(0, x)).empty())
        return "<x_0.0><y_0.0>" + raw;     // the leading words become the first body
    return raw.substr(x);
}

// Parse `<x_..><y_..>body<x_..><y_..><class_X>` elements in document order.
// `dedupe` drops exact repeats (same class, box and body), which the model emits
// fairly often on dense pages.
inline void parse_blocks(const std::string & raw_in, std::vector<Block> & blocks,
                         bool dedupe = true) {
    using namespace markup_detail;
    blocks.clear();
    const std::string raw = repair_empty_prefix(raw_in);
    size_t cursor = 0;
    for (;;) {
        const size_t open = raw.find("<x_", cursor);
        if (open == std::string::npos) break;
        Block b;
        size_t p = 0;
        if (!read_coord(raw, open, "<x_", b.x1, p)) { cursor = open + 3; continue; }
        if (!read_coord(raw, p, "<y_", b.y1, p)) { cursor = open + 3; continue; }
        const size_t body_start = p;
        // non-greedy: the first position that closes the element
        size_t j = body_start;
        bool found = false;
        size_t after = 0;
        while ((j = raw.find("<x_", j)) != std::string::npos) {
            if (read_close(raw, j, b.x2, b.y2, b.cls, after)) { found = true; break; }
            j += 3;
        }
        if (!found) break;
        b.text = raw.substr(body_start, j - body_start);
        if (!dedupe) {
            blocks.push_back(std::move(b));
        } else {
            bool dup = false;
            for (const auto & e : blocks) {
                if (e.cls == b.cls && e.x1 == b.x1 && e.y1 == b.y1 && e.x2 == b.x2 &&
                    e.y2 == b.y2 && e.text == b.text) { dup = true; break; }
            }
            if (!dup) blocks.push_back(std::move(b));
        }
        cursor = after;
    }
}

// One block's rendered content: `<tbc>` dropped, LaTeX tables turned into HTML.
// Upstream keeps the literal body in `text` and the converted one in `content`.
inline std::string block_content(const Block & b) {
    std::string t = b.text;
    t = markup_detail::replace_all(t, "<tbc>", "");
    t = markup_detail::replace_all(t, "\\<|unk|\\>", "");
    t = markup_detail::replace_all(t, "\\unknown", "");
    t = markup_detail::trim(t);
    if (b.cls == "Table" || b.cls == "Chart") {
        std::string html;
        if (markup_detail::latex_table_to_html(t, html)) {
            const size_t at = t.find("\\begin{tabular}");
            const size_t en = t.find("\\end{tabular}");
            if (at != std::string::npos && en != std::string::npos)
                t = t.substr(0, at) + html + t.substr(en + strlen("\\end{tabular}"));
            else
                t = html;
        }
    }
    return t;
}

// Upstream mmd_to_markdown: Title -> `#`, Section-header -> `##`, everything else
// verbatim; blocks joined with a blank line and nothing re-sorted.
inline std::string to_markdown(const std::vector<Block> & blocks) {
    std::string out;
    for (const auto & b : blocks) {
        std::string t = block_content(b);
        if (t.empty()) continue;
        if (b.cls == "Title" && !markup_detail::starts_with(t, "#")) t = "# " + t;
        else if (b.cls == "Section-header" && !markup_detail::starts_with(t, "#")) t = "## " + t;
        if (!out.empty()) out += "\n\n";
        out += t;
    }
    return out;
}

inline std::string to_plain(const std::vector<Block> & blocks) {
    std::string out;
    for (const auto & b : blocks) {
        const std::string t = block_content(b);
        if (t.empty()) continue;
        if (!out.empty()) out += "\n\n";
        out += t;
    }
    return out;
}

// postprocess_document()-shaped JSON: normalized bbox + pixel bbox per block.
inline std::string to_json(const std::vector<Block> & blocks, int img_w, int img_h) {
    auto num = [](double v) {
        char b[40];
        snprintf(b, sizeof(b), "%.6g", v);
        return std::string(b);
    };
    std::string o = "{\"coordinate_frame\":\"content_normalized\"";
    if (img_w > 0 && img_h > 0) {
        o += ",\"image_size\":[" + std::to_string(img_w) + "," + std::to_string(img_h) + "]";
    }
    o += ",\"markdown\":\"" + markup_detail::json_escape(to_markdown(blocks)) + "\"";
    o += ",\"blocks\":[";
    bool first = true;
    for (const auto & b : blocks) {
        if (!first) o += ",";
        first = false;
        const std::string content = block_content(b);
        o += "{\"class\":\"" + markup_detail::json_escape(b.cls) + "\"";
        o += ",\"bbox\":[" + num(b.x1) + "," + num(b.y1) + "," + num(b.x2) + "," + num(b.y2) + "]";
        if (img_w > 0 && img_h > 0) {
            // upstream transform_bbox_to_original: sort the corners, scale, clamp
            auto px = [](double v, int size) {
                return std::min((double)size, std::max(0.0, v * (double)size));
            };
            const double x1 = px(std::min(b.x1, b.x2), img_w);
            const double x2 = px(std::max(b.x1, b.x2), img_w);
            const double y1 = px(std::min(b.y1, b.y2), img_h);
            const double y2 = px(std::max(b.y1, b.y2), img_h);
            o += ",\"bbox_pixels\":[" + num(x1) + "," + num(y1) + "," + num(x2) + "," +
                 num(y2) + "]";
        }
        o += ",\"text\":\"" + markup_detail::json_escape(markup_detail::trim(b.text)) + "\"";
        o += ",\"content\":\"" + markup_detail::json_escape(content) + "\"";
        o += "}";
    }
    o += "],\"warnings\":[";
    first = true;
    for (size_t i = 0; i < blocks.size(); i++) {
        if (blocks[i].cls != "Table") continue;
        if (block_content(blocks[i]).find("\\begin{tabular}") == std::string::npos) continue;
        if (!first) o += ",";
        first = false;
        o += "{\"block_index\":" + std::to_string(i) +
             ",\"message\":\"Table markup could not be fully converted; original content "
             "preserved.\"}";
    }
    o += "]}";
    return o;
}

}  // namespace nemotron_parse
