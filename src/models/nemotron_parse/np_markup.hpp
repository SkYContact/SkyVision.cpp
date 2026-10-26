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
// Covered, matching upstream: block parsing + exact dedupe, `<tbc>` cleanup,
// text typography (soft hyphens, List-item bullets - engine/text_cleanup.py),
// Title/Section-header markdown, LaTeX `tabular` -> HTML, chart bodies (pipe
// table -> HTML, `title:` -> <caption>, `subtitle:` appended to it -
// engine/chart_caption.py + engine/chart_context.py), caption relocation
// (engine/caption_relocation.py) and `raw_out` in the JSON.
//
// Deliberate simplifications (all presentation-only, none change the block/bbox
// contract):
//   * header promotion uses the shape heuristic only (upstream also wants the
//     first row to be mostly bold, which LaTeX tabular markup rarely carries)
//   * chart panels/multi-series rendering is not re-implemented; the body's DSL
//     lines (`axis[...]`, `bind:`) are kept as upstream keeps them
//   * upstream's `merge.py` is a different feature, not a postprocess stage: it
//     merges a direct page prediction with a separate detector/crop prediction
//     source (IoU+Jaccard dedupe, extras inserted in reading order). It needs a
//     second model and JSONL plumbing; for a single prediction it keeps the
//     model's order, which is what this renderer does.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
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

// ---------------------------------------------------------------------------
// Chart bodies. A Chart block answers with a small DSL plus a pipe table:
//
//   <chart type="bar">
//   title: Revenue by region in 2026
//   axis[y]: millions USD
//   bind: millions USD ;  ; y
//
//   | Region | millions USD |
//   | --- | --- |
//   | North | 35 |
//   </chart>
//
// Upstream (engine/chart_caption.py) renders the pipe table as HTML and moves the
// `title:` line into its `<caption>`, removing the line from the body; everything
// else (the tags, `axis[...]`, `bind:`) is left as-is, so that is what happens
// here. The table is only converted when it is a plain pipe table: no escaped
// pipes, no backticks, at least three rows, equal widths, a separator row.

inline bool split_lines(const std::string & s, std::vector<std::string> & out) {
    out.clear();
    size_t p = 0;
    while (p <= s.size()) {
        const size_t e = s.find('\n', p);
        if (e == std::string::npos) {
            out.push_back(s.substr(p));
            break;
        }
        out.push_back(s.substr(p, e - p));
        p = e + 1;
    }
    return !out.empty();
}

inline bool is_table_row(const std::string & line) {
    const std::string t = trim(line);
    return t.size() >= 2 && t.front() == '|' && t.back() == '|';
}

inline bool is_separator_row(const std::string & line) {
    if (!is_table_row(line)) return false;
    const std::string t = trim(line);
    bool seen = false;
    for (size_t i = 1; i + 1 < t.size(); i++) {
        const char c = t[i];
        if (c == '|' || c == '-' || c == ':' || c == ' ' || c == '\t') {
            if (c == '-') seen = true;
            continue;
        }
        return false;
    }
    return seen;
}

inline std::vector<std::string> table_cells(const std::string & line) {
    std::vector<std::string> cells;
    const std::string t = trim(line);
    size_t start = 1;                       // skip the leading '|'
    for (size_t i = 1; i < t.size(); i++) {
        if (t[i] == '|') {
            cells.push_back(trim(t.substr(start, i - start)));
            start = i + 1;
        }
    }
    return cells;
}

inline std::string html_escape_attr(const std::string & s) { return html_escape(s); }

// pipe table -> <table>, with `caption` as its <caption> when non-empty
inline bool pipe_table_to_html(const std::vector<std::string> & rows, const std::string & caption,
                               std::string & html) {
    if (rows.size() < 3) return false;
    std::vector<std::vector<std::string>> cells;
    for (const auto & r : rows) {
        if (r.find("\\|") != std::string::npos || r.find('`') != std::string::npos) return false;
        cells.push_back(table_cells(r));
    }
    const size_t cols = cells[0].size();
    if (cols == 0) return false;
    for (size_t i = 1; i < cells.size(); i++)
        if (cells[i].size() != cols) return false;
    if (!is_separator_row(rows[1])) return false;
    for (size_t c = 0; c < cols; c++) {
        std::string sep = cells[1][c];
        sep.erase(std::remove_if(sep.begin(), sep.end(),
                                 [](char ch) { return ch == ' ' || ch == '\t'; }), sep.end());
        if (sep.empty()) return false;
        for (char ch : sep)
            if (ch != '-' && ch != ':') return false;
    }
    std::string o = "<table>";
    if (!caption.empty())
        o += "<caption>" + html_escape_attr(caption) + "</caption>";
    o += "<thead><tr>";
    for (size_t c = 0; c < cols; c++)
        o += "<th scope=\"col\">" + html_escape_attr(cells[0][c]) + "</th>";
    o += "</tr></thead><tbody>";
    for (size_t r = 2; r < cells.size(); r++) {
        o += "<tr>";
        for (size_t c = 0; c < cols; c++)
            o += "<td>" + html_escape_attr(cells[r][c]) + "</td>";
        o += "</tr>";
    }
    o += "</tbody></table>";
    html = o;
    return true;
}

// One `<chart ...> ... </chart>` run: pull `title:` out, render its pipe table,
// drop the title line. Returns the transformed body and how many tables changed.
inline std::string render_chart(const std::string & body) {
    const size_t open = body.find("<chart");
    if (open == std::string::npos) return body;
    const size_t gt = body.find('>', open);
    if (gt == std::string::npos) return body;
    const size_t close = body.find("</chart>", gt);
    if (close == std::string::npos) return body;
    const std::string inner = body.substr(gt + 1, close - gt - 1);

    std::vector<std::string> lines;
    split_lines(inner, lines);
    std::string title;
    std::vector<std::string> subtitles;
    for (const auto & l : lines) {
        const std::string t = trim(l);
        if (title.empty() && t.compare(0, 6, "title:") == 0) {
            title = trim(t.substr(6));
            continue;
        }
        // engine/chart_context.py: explicit `subtitle:` values are preserved by
        // appending them to the table caption (only when a caption exists)
        if (t.compare(0, 9, "subtitle:") == 0) {
            const std::string v = trim(t.substr(9));
            if (!v.empty()) subtitles.push_back(v);
        }
    }
    std::string caption = title;
    if (!caption.empty() && !subtitles.empty()) {
        std::string extra;
        for (const auto & s : subtitles) {
            if (caption.find(s) != std::string::npos || extra.find(s) != std::string::npos) continue;
            extra += " — " + s;
        }
        caption += extra;
    }

    // locate the first plain pipe table
    bool converted = false;
    for (size_t i = 0; i + 2 < lines.size(); i++) {
        if (!is_table_row(lines[i]) || !is_separator_row(lines[i + 1])) continue;
        size_t end = i + 2;
        while (end < lines.size() && is_table_row(lines[end])) end++;
        const std::vector<std::string> rows(lines.begin() + (long)i, lines.begin() + (long)end);
        std::string html;
        if (!pipe_table_to_html(rows, caption, html)) continue;
        lines.erase(lines.begin() + (long)i, lines.begin() + (long)end);
        lines.insert(lines.begin() + (long)i, html);
        converted = true;
        break;
    }
    if (!converted) return body;              // no table converted: leave the chart alone
    // the title now lives in the table's <caption>
    for (size_t i = 0; i < lines.size(); i++) {
        if (trim(lines[i]).compare(0, 6, "title:") == 0) {
            lines.erase(lines.begin() + (long)i);
            break;
        }
    }
    std::string out;
    for (const auto & l : lines) out += l + "\n";
    return body.substr(0, gt + 1) + out + body.substr(close);
}

}  // namespace markup_detail

// ---------------------------------------------------------------------------
// Text-only typography (upstream engine/text_cleanup.py).
//
//   * soft hyphens (U+00AD) between two word characters are dropped, for the
//     text-like roles only, and never inside code/maths (backtick, `$`, `\(`,
//     `\[`)
//   * List-item bullets (`•◦▪●‣⁃–—`) at the start of the line become `- `, which
//     is what makes upstream's lists render as Markdown lists
namespace markup_detail {

inline bool text_role(const std::string & cls) {
    static const char * const kRoles[] = { "Text", "List-item", "Caption", "Section-header",
                                           "Page-header", "Page-footer", "Footnote", "Title",
                                           nullptr };
    for (int i = 0; kRoles[i]; i++)
        if (cls == kRoles[i]) return true;
    return false;
}

inline bool word_byte(unsigned char c) {
    if (c == '_') return false;
    if (c >= 0x80) return true;                 // part of a multi-byte letter
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

// U+00AD is 0xC2 0xAD in UTF-8
inline std::string drop_soft_hyphens(const std::string & t) {
    std::string o;
    o.reserve(t.size());
    for (size_t i = 0; i < t.size();) {
        if (i + 2 < t.size() && (unsigned char)t[i] == 0xC2 && (unsigned char)t[i + 1] == 0xAD &&
            !o.empty() && word_byte((unsigned char)o.back()) &&
            word_byte((unsigned char)t[i + 2])) {
            i += 2;
            continue;
        }
        o += t[i++];
    }
    return o;
}

// leading "<spaces><bullet><spaces>" -> "- "
inline std::string normalize_bullet(const std::string & t) {
    static const char * const kBulls[] = { "\xE2\x80\xA2", "\xE2\x97\xA6", "\xE2\x96\xAA",
                                           "\xE2\x97\x8F", "\xE2\x80\xA3", "\xE2\x81\x83",
                                           "\xE2\x80\x93", "\xE2\x80\x94", nullptr };
    size_t p = 0;
    while (p < t.size() && (t[p] == ' ' || t[p] == '\t')) p++;
    for (int i = 0; kBulls[i]; i++) {
        const size_t n = strlen(kBulls[i]);
        if (t.compare(p, n, kBulls[i]) != 0) continue;
        size_t q = p + n;
        size_t spaces = 0;
        while (q < t.size() && (t[q] == ' ' || t[q] == '\t')) {
            q++;
            spaces++;
        }
        if (!spaces) return t;                   // a bare bullet is not a bullet line
        return "- " + t.substr(q);
    }
    return t;
}

inline std::string normalize_text(const std::string & cls, const std::string & text) {
    std::string t = text;
    const bool codeish = t.find('`') != std::string::npos || t.find('$') != std::string::npos ||
                         t.find("\\(") != std::string::npos ||
                         t.find("\\[") != std::string::npos;
    if (text_role(cls) && !codeish) t = drop_soft_hyphens(t);
    if (cls == "List-item") t = normalize_bullet(t);
    return t;
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
        // an element with no text at all is a parsing artifact (upstream drops it
        // in repair_empty_prefix); keep the block list clean for the JSON view too
        if (trim(replace_all(b.text, "<tbc>", "")).empty()) { cursor = after; continue; }
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

// One block's rendered content: `<tbc>` dropped, LaTeX tables and chart pipe
// tables turned into HTML. Upstream keeps the literal body in `text` and the
// converted one in `content`.
inline std::string block_content(const Block & b) {
    std::string t = b.text;
    t = markup_detail::replace_all(t, "<tbc>", "");
    t = markup_detail::replace_all(t, "\\<|unk|\\>", "");
    t = markup_detail::replace_all(t, "\\unknown", "");
    t = markup_detail::trim(t);
    t = markup_detail::normalize_text(b.cls, t);
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
    if (t.find("<chart") != std::string::npos) t = markup_detail::render_chart(t);
    return t;
}

// ---------------------------------------------------------------------------
// Caption relocation (upstream engine/caption_relocation.py).
//
// A *plain* one-line Caption that sits immediately ABOVE an adjacent Chart, with
// >=80% horizontal overlap and a vertical gap in [0, 0.03] (normalised page
// units), is folded into that chart's `title:` and the Caption block disappears.
// Notes below the object stay where they are. Ties (two equally close candidates)
// are dropped rather than guessed. Everything is verified at the end: the text
// must actually survive inside a rendered `<caption>`.
namespace markup_detail {

// A caption qualifies only if it is plain single-line text: no emphasis/code
// markers, no `_italic_`, no HTML tag, no LaTeX command, no link tail.
inline bool plain_caption(const std::string & src, std::string & out) {
    std::string t = replace_all(src, "<tbc>", "");
    t = trim(t);
    if (t.empty() || t.find('\n') != std::string::npos) return false;
    for (const char * m : { "**", "__", "~~", "`", "_" })
        if (t.find(m) != std::string::npos) return false;   // conservative on `_`
    for (size_t i = 0; i + 1 < t.size(); i++) {
        if (t[i] == '<') {
            const char c = t[i + 1];
            if (isalpha((unsigned char)c) || c == '/' || c == '!') return false;
        }
        if (t[i] == '\\') {
            const char c = t[i + 1];
            if (isalpha((unsigned char)c) || c == '&' || c == '%' || c == '_') return false;
        }
    }
    if (t.find("](") != std::string::npos) return false;
    out = t;
    return true;
}

// Inject `label` as the chart's title: append to an existing `title:` line, else
// insert one right after the opening `<chart ...>` tag.
inline std::string chart_with_title(const std::string & body, const std::string & label) {
    std::vector<std::string> lines;
    split_lines(body, lines);
    for (auto & l : lines) {
        const std::string t = trim(l);
        if (t.compare(0, 6, "title:") == 0) {
            std::string cur = trim(t.substr(6));
            if (cur.find(label) != std::string::npos) return body;   // already there
            const size_t at = l.find("title:");
            l = l.substr(0, at) + "title: " + cur + " — " + label;
            std::string out;
            for (const auto & x : lines) out += x + "\n";
            return out;
        }
    }
    // no title line: insert after the opening tag
    const size_t open = body.find("<chart");
    const size_t gt = open == std::string::npos ? std::string::npos : body.find('>', open);
    if (gt == std::string::npos) return body;
    return body.substr(0, gt + 1) + "\ntitle: " + label + body.substr(gt + 1);
}

inline bool contains_caption(const std::string & rendered, const std::string & label) {
    size_t p = 0;
    while ((p = rendered.find("<caption>", p)) != std::string::npos) {
        const size_t e = rendered.find("</caption>", p);
        if (e == std::string::npos) return false;
        if (rendered.substr(p, e - p).find(label) != std::string::npos) return true;
        p = e + 1;
    }
    return false;
}

}  // namespace markup_detail

// Rewrites `blocks` in place: relocated Caption blocks are erased, their text
// merged into the target Chart block's body.
inline void relocate_captions(std::vector<Block> & blocks) {
    using namespace markup_detail;
    // pass 1: which chart could take which caption, with its vertical gap
    std::map<size_t, std::vector<std::pair<double, size_t>>> eligible;   // chart -> (gap, caption)
    for (size_t i = 0; i < blocks.size(); i++) {
        const Block & c = blocks[i];
        if (c.cls != "Caption") continue;
        std::string label;
        if (!plain_caption(c.text, label)) continue;
        std::vector<std::pair<double, size_t>> cands;
        for (int d = -1; d <= 1; d += 2) {           // adjacent blocks only
            const long j = (long)i + d;
            if (j < 0 || j >= (long)blocks.size()) continue;
            const Block & t = blocks[(size_t)j];
            if (t.cls != "Chart") continue;
            if (c.y2 > t.y1) continue;               // must sit fully above the chart
            const double w = std::min(c.x2 - c.x1, t.x2 - t.x1);
            const double overlap = std::max(0.0, std::min(c.x2, t.x2) - std::max(c.x1, t.x1));
            const double gap = std::max(t.y1 - c.y2, c.y1 - t.y2);
            if (w <= 0.0 || overlap / w < 0.8) continue;
            if (gap < 0.0 || gap > 0.03) continue;
            cands.push_back({ gap, (size_t)j });
        }
        if (cands.empty()) continue;
        std::sort(cands.begin(), cands.end());
        // two charts equally close: ambiguous, leave the caption alone
        if (cands.size() > 1 && std::fabs(cands[0].first - cands[1].first) < 1e-9) continue;
        eligible[cands[0].second].push_back({ cands[0].first, i });
    }

    // pass 2: one caption per chart, nearest wins, ties dropped, then verify that
    // the text really lands in a rendered <caption>
    std::vector<size_t> erase;
    for (auto & kv : eligible) {
        std::vector<std::pair<double, size_t>> caps = kv.second;
        std::sort(caps.begin(), caps.end());
        if (caps.size() > 1 && std::fabs(caps[0].first - caps[1].first) < 1e-9) continue;
        const size_t ci = caps[0].second;
        Block & chart = blocks[kv.first];
        std::string label;
        if (!plain_caption(blocks[ci].text, label)) continue;
        if (chart.text.find("<chart") == std::string::npos) continue;
        const std::string derived = chart_with_title(chart.text, label);
        if (derived == chart.text) continue;
        Block probe;
        probe.cls = chart.cls;
        probe.text = derived;
        if (!contains_caption(block_content(probe), label)) continue;   // did not survive
        chart.text = derived;
        erase.push_back(ci);
    }
    std::sort(erase.begin(), erase.end());
    erase.erase(std::unique(erase.begin(), erase.end()), erase.end());
    for (size_t i = erase.size(); i-- > 0;) blocks.erase(blocks.begin() + (long)erase[i]);
}

// Upstream mmd_to_markdown: Title -> `#`, Section-header -> `##`, everything else
// verbatim; blocks joined with a blank line and nothing re-sorted.
inline std::string to_markdown(const std::vector<Block> & blocks_in) {
    std::vector<Block> blocks = blocks_in;
    relocate_captions(blocks);
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

inline std::string to_plain(const std::vector<Block> & blocks_in) {
    std::vector<Block> blocks = blocks_in;
    relocate_captions(blocks);
    std::string out;
    for (const auto & b : blocks) {
        const std::string t = block_content(b);
        if (t.empty()) continue;
        if (!out.empty()) out += "\n\n";
        out += t;
    }
    return out;
}

// postprocess_document()-shaped JSON: normalized bbox + pixel bbox per block,
// plus the untouched model output in `raw_out` (as upstream does).
inline std::string to_json(const std::vector<Block> & blocks_in, int img_w, int img_h,
                           const std::string & raw_out = std::string()) {
    std::vector<Block> blocks = blocks_in;
    relocate_captions(blocks);
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
    if (!raw_out.empty())
        o += ",\"raw_out\":\"" + markup_detail::json_escape(raw_out) + "\"";
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
