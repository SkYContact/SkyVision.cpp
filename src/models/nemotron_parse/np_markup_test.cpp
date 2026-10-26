// np_markup_test - checks for the Nemotron-3.5-Parse answer renderer.
//
// np_markup.hpp is header-only and depends on nothing but the C++ standard
// library, so this target builds without ggml and runs anywhere (CI included).
//
//   cmake --build build --target np_markup_test && ./build/np_markup_test
//
// Non-ASCII characters are written as UTF-8 byte escapes on purpose: the source
// then compiles the same way under MSVC's default codepage and under gcc.
#include "np_markup.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace nemotron_parse;

static int failures = 0;

static void check(bool ok, const char * what) {
    printf("%s  %s\n", ok ? "[ok]  " : "[FAIL]", what);
    if (!ok) failures++;
}

static bool has(const std::string & hay, const std::string & needle) {
    return hay.find(needle) != std::string::npos;
}

static bool has(const std::string & hay, const char * needle) {
    return hay.find(needle) != std::string::npos;
}

// UTF-8 helpers for the typography checks
static const char * kSoftHyphen = "\xC2\xAD";       // U+00AD
static const char * kBulletDot = "\xE2\x80\xA2";    // U+2022
static const char * kEnDash = "\xE2\x80\x93";       // U+2013

int main() {
    // ---- chart blocks -----------------------------------------------------
    // the shape the model actually emits for a bar chart
    const std::string chart_raw =
        "<x_0.0781><y_0.0727><chart type=\"bar\">\n"
        "title: Revenue by region in 2026\n"
        "axis[y]: millions USD\n"
        "bind: millions USD ;  ; y\n"
        "\n"
        "| Region | millions USD |\n"
        "| --- | --- |\n"
        "| North | 35 |\n"
        "| South | 62 |\n"
        "</chart><x_0.9531><y_0.6078><class_Chart>\n";
    {
        std::vector<Block> b;
        parse_blocks(chart_raw, b);
        check(b.size() == 1 && b[0].cls == "Chart", "chart: one Chart block");
        const std::string md = to_markdown(b);
        check(has(md, "<caption>Revenue by region in 2026</caption>"),
              "chart: title became <caption>");
        check(has(md, "<th scope=\"col\">Region</th>"), "chart: header row rendered");
        check(has(md, "<td>South</td>"), "chart: body cell rendered");
        check(!has(md, "title:"), "chart: title line consumed");
        check(has(md, "axis[y]: millions USD"), "chart: other DSL lines kept (as upstream)");
    }

    // subtitle values are appended to the caption (engine/chart_context.py)
    {
        std::vector<Block> b;
        parse_blocks("<x_0.1><y_0.1><chart type=\"line\">\n"
                     "title: Load over time\n"
                     "subtitle: per region\n"
                     "subtitle: weekly\n"
                     "| t | v |\n| --- | --- |\n| 1 | 2 |\n| 2 | 3 |\n"
                     "</chart><x_0.5><y_0.5><class_Chart>", b);
        const std::string md = to_markdown(b);
        check(has(md, "<caption>Load over time \xE2\x80\x94 per region \xE2\x80\x94 weekly</caption>"),
              "chart: subtitles appended to the caption");
    }

    // ---- caption relocation (engine/caption_relocation.py) ----------------
    const std::string cap_above =
        "<x_0.10><y_0.10>Figure 1: yearly totals<x_0.60><y_0.13><class_Caption>"
        "<x_0.09><y_0.14><chart type=\"bar\">\n"
        "axis[y]: units\n"
        "| Region | v |\n| --- | --- |\n| North | 3 |\n| South | 4 |\n"
        "</chart><x_0.90><y_0.60><class_Chart>";
    {
        std::vector<Block> b;
        parse_blocks(cap_above, b);
        check(b.size() == 2, "relocate: two blocks before");
        const std::string md = to_markdown(b);
        check(has(md, "<caption>Figure 1: yearly totals</caption>"),
              "relocate: caption became the table caption");
        check(!has(md, "title:"), "relocate: title not left in markdown");
        check(!has(md, "<x_0.10>"), "relocate: caption block removed");
    }

    // a note BELOW the chart stays a standalone block
    {
        std::vector<Block> b;
        parse_blocks("<x_0.09><y_0.10><chart type=\"bar\">\n"
                     "| Region | v |\n| --- | --- |\n| North | 3 |\n| South | 4 |\n"
                     "</chart><x_0.90><y_0.60><class_Chart>"
                     "<x_0.10><y_0.62>Note: values are provisional<x_0.60><y_0.65><class_Caption>",
                     b);
        check(b.size() == 2, "note below: both blocks kept");
        const std::string md = to_markdown(b);
        check(has(md, "Note: values are provisional"), "note below: still rendered");
        check(!has(md, "title: Note"), "note below: not folded into the title");
    }

    // rich-text captions are left alone
    {
        std::vector<Block> b;
        parse_blocks("<x_0.10><y_0.10>**Figure 1** totals<x_0.60><y_0.13><class_Caption>"
                     "<x_0.09><y_0.14><chart type=\"bar\">\n"
                     "| Region | v |\n| --- | --- |\n| North | 3 |\n| South | 4 |\n"
                     "</chart><x_0.90><y_0.60><class_Chart>", b);
        const std::string md = to_markdown(b);
        check(has(md, "**Figure 1** totals"), "rich caption: kept as its own block");
        check(has(md, "<table>"), "rich caption: table still rendered");
    }

    // ---- tables -----------------------------------------------------------
    {
        std::vector<Block> b;
        parse_blocks("<x_0.1><y_0.1>Report<x_0.5><y_0.12><class_Text>"
                     "<x_0.1><y_0.2>\\begin{tabular}{cc}\nA & B \\\\\n1 & 2 \\\\\n\\end{tabular}"
                     "<x_0.5><y_0.4><class_Table>", b);
        const std::string md = to_markdown(b);
        check(has(md, "<table>") && has(md, "<td>1</td>"), "latex table: converted");
    }

    // ---- typography (engine/text_cleanup.py) ------------------------------
    {
        std::vector<Block> b;
        const std::string raw = std::string("<x_0.1><y_0.1>hy") + kSoftHyphen +
                                "phen word<x_0.5><y_0.12><class_Text>" +
                                "<x_0.1><y_0.2>a" + kSoftHyphen +
                                "b with `code`<x_0.5><y_0.22><class_Text>";
        parse_blocks(raw, b);
        check(b.size() == 2, "soft hyphen: two blocks");
        const std::string md = to_markdown(b);
        check(has(md, "hyphen word"), "soft hyphen: dropped in a text role");
        check(!has(md, std::string("hy") + kSoftHyphen + "phen"),
              "soft hyphen: character is gone from the text role");
        check(has(md, std::string("a") + kSoftHyphen + "b"), "soft hyphen: kept next to code");
    }
    {
        std::vector<Block> b;
        parse_blocks(std::string("<x_0.1><y_0.1>") + kBulletDot +
                     " first item<x_0.5><y_0.12><class_List-item>" + "<x_0.1><y_0.2>" + kEnDash +
                     " second item<x_0.5><y_0.22><class_List-item>" +
                     "<x_0.1><y_0.3>plain text<x_0.5><y_0.32><class_Text>", b);
        const std::string md = to_markdown(b);
        check(has(md, "- first item"), "bullet: U+2022 becomes '- '");
        check(has(md, "- second item"), "bullet: U+2013 becomes '- '");
        check(!has(md, "- plain text"), "bullet: Text blocks are untouched");
    }

    // ---- empty elements ---------------------------------------------------
    {
        std::vector<Block> b;
        parse_blocks("<x_0.1><y_0.1><x_0.2><y_0.12><class_Text>"
                     "<x_0.1><y_0.2>real text<x_0.5><y_0.22><class_Text>", b);
        check(b.size() == 1 && has(b[0].text, "real text"), "empty element: skipped");
    }

    // ---- JSON -------------------------------------------------------------
    {
        std::vector<Block> b;
        parse_blocks(chart_raw, b);
        const std::string js = to_json(b, 1000, 1300, chart_raw);
        check(has(js, "\"raw_out\":\"<x_0.0781>"), "json: raw_out echoed");
        check(has(js, "\"bbox_pixels\":"), "json: pixel bbox present");
        check(has(js, "\"markdown\":\""), "json: markdown present");
    }

    printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "all passed", failures,
           failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
