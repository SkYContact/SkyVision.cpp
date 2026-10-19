// json_scan.hpp - cheap key scans over a JSON *text* blob (no DOM).
//
// The HF side files (tokenizer.json, preprocessor_config.json, generation
// config) are megabytes of JSON of which a handful of scalars matter, so these
// two helpers scan for a key directly. Full parsing lives in core/text/json.hpp;
// these are the "read one string / one number out of a config" case.
#pragma once

#include <cstddef>
#include <string>

namespace vqa {

// Parse a JSON string literal starting at js[i] == '"'; returns the decoded
// content and advances i past the closing quote.
std::string json_parse_string(const std::string & js, size_t & i);

// Extract a top-level numeric / bool value for "key". Returns false if absent.
bool json_scan_number(const std::string & js, const char * key, double & out);

}  // namespace vqa
