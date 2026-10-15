// tokenizer.hpp - byte level BPE tokenizer shared by every backend.
//
// All supported models ship an HF `tokenizer.json` with a `vocab` + `merges`
// pair; they differ in two ways, both exposed as flags here:
//
//   use_pretokenizer  split the text with the Qwen2/Qwen3 regex before BPE
//                     (without it the whole string is one BPE sequence, which is
//                     what the moondream/starmie vocabulary expects)
//   use_added_tokens  match `<|...|>` style specials literally before the
//                     pre-tokenizer and skip them when decoding
//
// The Unicode class tables used by the pre-tokenizer live in unicode_ranges.inc
// (regenerate with scripts/gen_unicode_ranges.py).
#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace vqa {

enum class TokenizerMode {
    PlainGpt2,   // moondream / starmie: whole-string BPE, no specials
    Qwen2Regex,  // Qwen2/Qwen3 (VideoChat3): regex pre-tokenizer + added tokens
};

struct Tokenizer {
    std::unordered_map<std::string, int> vocab;
    std::vector<std::string> id_to_piece;
    std::map<std::pair<std::string, std::string>, int> merge_rank;
    std::vector<std::pair<std::string, int>> added;   // (content, id), longest first
    std::vector<bool> is_added;

    bool use_pretokenizer = false;
    bool use_added_tokens = false;

    bool load(const char * tokenizer_json, TokenizerMode mode = TokenizerMode::PlainGpt2);

    std::vector<int> encode(const std::string & text) const;
    std::string decode(const std::vector<int> & ids) const;

    int id_of(const std::string & piece) const;
    const std::string & piece_of(int id) const;
    int size() const { return (int)id_to_piece.size(); }

  private:
    void encode_piece(const std::string & piece, std::vector<int> & out) const;
    int match_added(const std::string & text, size_t pos, size_t & len) const;
};

// CLI self test: tokenize + decode a raw string with the given mode.
bool debug_tokenize(const char * tokenizer_json, const std::string & text,
                    std::vector<int> & ids, std::string & decoded,
                    TokenizerMode mode = TokenizerMode::Qwen2Regex);

}  // namespace vqa
