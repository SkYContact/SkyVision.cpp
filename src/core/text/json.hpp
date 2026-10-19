// json.hpp - the smallest JSON value/parser the protocol front ends need.
//
// The engine deliberately carries no third-party JSON library: the CLI and the
// Web UI never parse structured input (they take flags or query strings), and
// the only consumer of real JSON is the MCP server (JSON-RPC over stdio) plus
// the tokenizer's config scan, which has its own purpose-built helpers in
// core/text/json_scan.hpp. Keeping one recursive-descent parser here instead of
// pulling a dependency in matches the rest of src/core (header-only, no ggml,
// no I/O).
#pragma once

#include "core/text/utf8.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace vqa {
namespace json {

struct Value {
    enum Type { Null, Bool, Number, String, Array, Object };

    Type type = Null;
    bool boolean = false;
    double number = 0.0;
    std::string text;
    std::vector<Value> items;                             // Array
    std::vector<std::pair<std::string, Value>> members;   // Object (insertion order)

    bool is_null() const { return type == Null; }
    bool is_bool() const { return type == Bool; }
    bool is_number() const { return type == Number; }
    bool is_string() const { return type == String; }
    bool is_array() const { return type == Array; }
    bool is_object() const { return type == Object; }

    // ---- object accessors (never throw; absent keys yield the default) ----
    const Value * get(const char * key) const {
        if (type != Object) return nullptr;
        for (const auto & m : members)
            if (m.first == key) return &m.second;
        return nullptr;
    }
    bool has(const char * key) const { return get(key) != nullptr; }

    std::string str(const char * key, const std::string & def = std::string()) const {
        const Value * v = get(key);
        return (v && v->is_string()) ? v->text : def;
    }
    // numbers given as strings are common in hand-written tool calls ("n": "64")
    double num(const char * key, double def) const {
        const Value * v = get(key);
        if (!v) return def;
        if (v->is_number()) return v->number;
        if (v->is_string() && !v->text.empty()) return atof(v->text.c_str());
        return def;
    }
    int integer(const char * key, int def) const { return (int)num(key, (double)def); }
    bool flag(const char * key, bool def) const {
        const Value * v = get(key);
        if (!v) return def;
        if (v->type == Bool) return v->boolean;
        if (v->is_number()) return v->number != 0.0;
        if (v->is_string()) return v->text == "true" || v->text == "1" || v->text == "yes";
        return def;
    }

    // ---- builders ----
    static Value make_object() { Value v; v.type = Object; return v; }
    static Value make_array() { Value v; v.type = Array; return v; }
    static Value make_string(const std::string & s) { Value v; v.type = String; v.text = s; return v; }
    static Value make_number(double d) { Value v; v.type = Number; v.number = d; return v; }
    static Value make_bool(bool b) { Value v; v.type = Bool; v.boolean = b; return v; }

    Value & set(const std::string & key, Value v) {
        type = Object;
        members.emplace_back(key, std::move(v));
        return *this;
    }
    Value & push(Value v) {
        type = Array;
        items.push_back(std::move(v));
        return *this;
    }
};

// ---------------------------------------------------------------- serialize
inline std::string escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) {
                    char b[8];
                    snprintf(b, sizeof(b), "\\u%04x", c);
                    out += b;
                } else {
                    out += (char)c;
                }
        }
    }
    return out;
}

inline std::string quote(const std::string & s) { return "\"" + escape(s) + "\""; }

inline std::string number_text(double v) {
    if (!std::isfinite(v)) return "null";
    char b[40];
    if (v == (double)(long long)v && std::fabs(v) < 1e15) {
        snprintf(b, sizeof(b), "%lld", (long long)v);
    } else {
        // 10 significant digits: enough for coordinates and probabilities, and
        // short enough that 0.3 does not print as 0.29999999999999999
        snprintf(b, sizeof(b), "%.10g", v);
    }
    return b;
}

inline void dump_to(std::string & out, const Value & v) {
    switch (v.type) {
        case Value::Null: out += "null"; break;
        case Value::Bool: out += v.boolean ? "true" : "false"; break;
        case Value::Number: out += number_text(v.number); break;
        case Value::String: out += quote(v.text); break;
        case Value::Array:
            out += '[';
            for (size_t i = 0; i < v.items.size(); i++) {
                if (i) out += ',';
                dump_to(out, v.items[i]);
            }
            out += ']';
            break;
        case Value::Object:
            out += '{';
            for (size_t i = 0; i < v.members.size(); i++) {
                if (i) out += ',';
                out += quote(v.members[i].first);
                out += ':';
                dump_to(out, v.members[i].second);
            }
            out += '}';
            break;
    }
}

inline std::string dump(const Value & v) {
    std::string out;
    dump_to(out, v);
    return out;
}

// ---------------------------------------------------------------- parse
namespace detail {

struct Parser {
    const char * s = nullptr;
    size_t n = 0, i = 0;
    std::string err;

    explicit Parser(const std::string & text) : s(text.data()), n(text.size()) {}

    void ws() {
        while (i < n) {
            const char c = s[i];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') i++;
            else break;
        }
    }
    bool fail(const char * m) {
        if (err.empty()) {
            err = m;
            err += " at offset ";
            err += std::to_string(i);
        }
        return false;
    }
    static bool is_digit(char c) { return c >= '0' && c <= '9'; }

    bool literal(const char * w) {
        const size_t k = strlen(w);
        if (i + k > n || memcmp(s + i, w, k) != 0) return fail("unexpected token");
        i += k;
        return true;
    }

    bool number(Value & v) {
        const size_t start = i;
        if (i < n && (s[i] == '-' || s[i] == '+')) i++;
        bool digits = false;
        while (i < n && is_digit(s[i])) { i++; digits = true; }
        if (i < n && s[i] == '.') {
            i++;
            while (i < n && is_digit(s[i])) { i++; digits = true; }
        }
        if (!digits) return fail("malformed number");
        if (i < n && (s[i] == 'e' || s[i] == 'E')) {
            i++;
            if (i < n && (s[i] == '-' || s[i] == '+')) i++;
            int ed = 0;
            while (i < n && is_digit(s[i])) { i++; ed++; }
            if (!ed) return fail("malformed exponent");
        }
        v.type = Value::Number;
        v.number = strtod(std::string(s + start, i - start).c_str(), nullptr);
        return true;
    }

    bool string(std::string & out) {
        if (i >= n || s[i] != '"') return fail("expected string");
        i++;
        while (true) {
            if (i >= n) return fail("unterminated string");
            const unsigned char c = (unsigned char)s[i++];
            if (c == '"') return true;
            if (c < 0x20) return fail("control character in string");
            if (c != '\\') { out += (char)c; continue; }
            if (i >= n) return fail("unterminated escape");
            const char e = s[i++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!hex4(cp)) return false;
                    if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < n && s[i] == '\\' && s[i + 1] == 'u') {
                        i += 2;
                        uint32_t lo = 0;
                        if (!hex4(lo)) return false;
                        if (lo >= 0xDC00 && lo <= 0xDFFF)
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        else
                            return fail("malformed surrogate pair");
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: return fail("unknown escape");
            }
        }
    }

    bool hex4(uint32_t & out) {
        if (i + 4 > n) return fail("truncated \\u escape");
        out = 0;
        for (int k = 0; k < 4; k++) {
            const char c = s[i++];
            out <<= 4;
            if (c >= '0' && c <= '9') out |= (uint32_t)(c - '0');
            else if (c >= 'a' && c <= 'f') out |= (uint32_t)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') out |= (uint32_t)(c - 'A' + 10);
            else return fail("bad hex digit");
        }
        return true;
    }

    bool value(Value & v, int depth) {
        if (depth > 64) return fail("nesting too deep");
        ws();
        if (i >= n) return fail("unexpected end of input");
        switch (s[i]) {
            case '{': {
                v.type = Value::Object;
                i++;
                ws();
                if (i < n && s[i] == '}') { i++; return true; }
                while (true) {
                    ws();
                    std::string key;
                    if (!string(key)) return false;
                    ws();
                    if (i >= n || s[i] != ':') return fail("expected ':'");
                    i++;
                    Value child;
                    if (!value(child, depth + 1)) return false;
                    v.members.emplace_back(std::move(key), std::move(child));
                    ws();
                    if (i >= n) return fail("unterminated object");
                    if (s[i] == ',') { i++; continue; }
                    if (s[i] == '}') { i++; return true; }
                    return fail("expected ',' or '}'");
                }
            }
            case '[': {
                v.type = Value::Array;
                i++;
                ws();
                if (i < n && s[i] == ']') { i++; return true; }
                while (true) {
                    Value child;
                    if (!value(child, depth + 1)) return false;
                    v.items.push_back(std::move(child));
                    ws();
                    if (i >= n) return fail("unterminated array");
                    if (s[i] == ',') { i++; continue; }
                    if (s[i] == ']') { i++; return true; }
                    return fail("expected ',' or ']'");
                }
            }
            case '"':
                v.type = Value::String;
                return string(v.text);
            case 't':
                v.type = Value::Bool;
                v.boolean = true;
                return literal("true");
            case 'f':
                v.type = Value::Bool;
                v.boolean = false;
                return literal("false");
            case 'n':
                v.type = Value::Null;
                return literal("null");
            default:
                return number(v);
        }
    }
};

}  // namespace detail

// Parse one complete JSON document. On failure `err` carries offset + reason.
inline bool parse(const std::string & text, Value & out, std::string & err) {
    detail::Parser p(text);
    out = Value();
    if (!p.value(out, 0)) { err = p.err.empty() ? "parse error" : p.err; return false; }
    p.ws();
    if (p.i != p.n) { err = "trailing characters at offset " + std::to_string(p.i); return false; }
    return true;
}

}  // namespace json
}  // namespace vqa
