// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once
//
// Byte-level BPE tokenizer (GPT-2 / Qwen style) that loads a HuggingFace tokenizer.json.
// Pieces, mirroring the reference exactly:
//   1. pre-tokenization  - the tokenizer.json Split regex, run with PCRE2 in UTF+UCP mode
//                          so \p{L}, \p{N}, \p{M} and the (?i:) contractions behave correctly.
//   2. byte level        - each pretoken's raw UTF-8 bytes are mapped to the GPT-2 "printable"
//                          unicode alphabet (byte 0x20 -> U+0120 'Ġ', etc.).
//   3. BPE merges        - greedily merge the adjacent pair with the lowest merge rank.
//   4. vocab lookup      - merged symbol string -> token id.
//   5. special tokens    - added_tokens (e.g. <|im_start|>) are split out first and emitted
//                          as their ids directly.
//
#include <cstdint>
#include <climits>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <set>
#include <functional>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

namespace cppgrad {
namespace io {

class BPETokenizer {
public:
    explicit BPETokenizer(const std::string& tokenizer_json_path) {
        std::ifstream f(tokenizer_json_path);
        if (!f.is_open()) throw std::runtime_error("tokenizer: cannot open " + tokenizer_json_path);
        nlohmann::json j;
        f >> j;

        // vocab: token string -> id
        for (auto& [tok, id] : j["model"]["vocab"].items()) {
            int i = id.get<int>();
            vocab_[tok] = i;
            if ((int)id_to_tok_.size() <= i) id_to_tok_.resize(i + 1);
            id_to_tok_[i] = tok;
        }
        // merges: ranked adjacent pairs. Stored either as ["A B"] or ["A","B"].
        int rank = 0;
        for (auto& m : j["model"]["merges"]) {
            std::string a, b;
            if (m.is_array()) { a = m[0].get<std::string>(); b = m[1].get<std::string>(); }
            else { std::string s = m.get<std::string>(); auto sp = s.find(' '); a = s.substr(0, sp); b = s.substr(sp + 1); }
            merge_rank_[a + '\0' + b] = rank++;
        }
        // added/special tokens: content -> id (split out before BPE)
        if (j.contains("added_tokens")) {
            for (auto& t : j["added_tokens"]) {
                int id = t["id"].get<int>();
                std::string content = t["content"].get<std::string>();
                special_[content] = id;
                if ((int)id_to_tok_.size() <= id) id_to_tok_.resize(id + 1);
                id_to_tok_[id] = content;
                special_is_special_.insert(id);
            }
        }
        build_byte_maps();
        compile_regex(j["pre_tokenizer"]);
    }

    ~BPETokenizer() {
        if (re_) pcre2_code_free(re_);
    }

    std::vector<int32_t> encode(const std::string& text) const {
        std::vector<int32_t> ids;
        // Split on special-token contents (longest match wins), BPE the gaps.
        size_t i = 0;
        while (i < text.size()) {
            size_t next = std::string::npos, next_len = 0; int next_id = -1;
            for (auto& [content, id] : special_) {
                if (content.empty()) continue;
                size_t pos = text.find(content, i);
                if (pos != std::string::npos && (pos < next || (pos == next && content.size() > next_len))) {
                    next = pos; next_len = content.size(); next_id = id;
                }
            }
            size_t gap_end = (next == std::string::npos) ? text.size() : next;
            if (gap_end > i) encode_chunk(text.substr(i, gap_end - i), ids);
            if (next == std::string::npos) break;
            ids.push_back(next_id);
            i = next + next_len;
        }
        return ids;
    }

    std::string decode(const std::vector<int32_t>& ids, bool skip_special = false) const {
        std::string out;
        for (int32_t id : ids) {
            if (id < 0 || id >= (int)id_to_tok_.size()) continue;
            if (special_is_special_.count(id)) {
                if (!skip_special) out += id_to_tok_[id];  // special content is literal
                continue;
            }
            // byte-level token: map each unicode char back to its byte
            const std::string& tok = id_to_tok_[id];
            size_t p = 0;
            while (p < tok.size()) {
                uint32_t cp = utf8_next(tok, p);
                auto it = unicode_to_byte_.find(cp);
                if (it != unicode_to_byte_.end()) out.push_back((char)it->second);
            }
        }
        return out;
    }

    int vocab_size() const { return (int)id_to_tok_.size(); }

private:
    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> id_to_tok_;
    std::unordered_map<std::string, int> merge_rank_;     // "A\0B" -> rank
    std::unordered_map<std::string, int> special_;        // content -> id
    std::set<int> special_is_special_;
    std::string byte_to_tok_[256];                        // byte -> UTF-8 of its GPT-2 unicode char
    std::unordered_map<uint32_t, uint8_t> unicode_to_byte_;
    pcre2_code* re_ = nullptr;

    // GPT-2 bytes<->unicode: printable bytes map to themselves, the rest to 256+n.
    void build_byte_maps() {
        std::vector<int> bs, cs;
        for (int i = 33; i <= 126; i++) bs.push_back(i);
        for (int i = 161; i <= 172; i++) bs.push_back(i);
        for (int i = 174; i <= 255; i++) bs.push_back(i);
        cs = bs;
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) { bs.push_back(b); cs.push_back(256 + n); n++; }
        }
        for (size_t i = 0; i < bs.size(); i++) {
            byte_to_tok_[(uint8_t)bs[i]] = utf8_encode((uint32_t)cs[i]);
            unicode_to_byte_[(uint32_t)cs[i]] = (uint8_t)bs[i];
        }
    }

    void compile_regex(const nlohmann::json& pre) {
        // Find the Split regex pattern inside the pre_tokenizer (possibly a Sequence).
        std::string pattern;
        std::function<void(const nlohmann::json&)> find = [&](const nlohmann::json& p) {
            if (!pattern.empty()) return;
            if (p.contains("type") && p["type"] == "Split" && p.contains("pattern") && p["pattern"].contains("Regex"))
                pattern = p["pattern"]["Regex"].get<std::string>();
            if (p.contains("pretokenizers")) for (auto& s : p["pretokenizers"]) find(s);
        };
        find(pre);
        if (pattern.empty()) throw std::runtime_error("tokenizer: no Split regex in pre_tokenizer");

        int errcode; PCRE2_SIZE erroff;
        re_ = pcre2_compile((PCRE2_SPTR)pattern.c_str(), pattern.size(),
                            PCRE2_UTF | PCRE2_UCP, &errcode, &erroff, nullptr);
        if (!re_) {
            PCRE2_UCHAR buf[256]; pcre2_get_error_message(errcode, buf, sizeof(buf));
            throw std::runtime_error(std::string("tokenizer: regex compile failed: ") + (char*)buf);
        }
    }

    // BPE a single regex pretoken (the chunk between special tokens is first regex-split).
    void encode_chunk(const std::string& chunk, std::vector<int32_t>& ids) const {
        pcre2_match_data* md = pcre2_match_data_create_from_pattern(re_, nullptr);
        PCRE2_SIZE off = 0;
        while (off < chunk.size()) {
            int rc = pcre2_match(re_, (PCRE2_SPTR)chunk.c_str(), chunk.size(), off, 0, md, nullptr);
            if (rc < 0) break;
            PCRE2_SIZE* ov = pcre2_get_ovector_pointer(md);
            size_t ms = ov[0], me = ov[1];
            if (me <= ms) { off = me + 1; continue; }
            bpe_pretoken(chunk.substr(ms, me - ms), ids);
            off = me;
        }
        pcre2_match_data_free(md);
    }

    void bpe_pretoken(const std::string& piece, std::vector<int32_t>& ids) const {
        // Byte-level: each raw byte -> its GPT-2 unicode char (as a UTF-8 symbol).
        std::vector<std::string> sym;
        sym.reserve(piece.size());
        for (unsigned char c : piece) sym.push_back(byte_to_tok_[c]);
        if (sym.empty()) return;

        // Greedy BPE: repeatedly merge ALL occurrences of the globally lowest-rank pair.
        while (sym.size() >= 2) {
            int best_rank = INT32_MAX; size_t best_i = 0;
            for (size_t i = 0; i + 1 < sym.size(); i++) {
                auto it = merge_rank_.find(sym[i] + '\0' + sym[i + 1]);
                if (it != merge_rank_.end() && it->second < best_rank) { best_rank = it->second; best_i = i; }
            }
            if (best_rank == INT32_MAX) break;
            const std::string merged = sym[best_i] + sym[best_i + 1];
            std::vector<std::string> next;
            next.reserve(sym.size());
            for (size_t i = 0; i < sym.size();) {
                if (i + 1 < sym.size() && sym[i] == sym[best_i] && sym[i + 1] == sym[best_i + 1]) {
                    next.push_back(merged); i += 2;
                } else { next.push_back(sym[i]); i += 1; }
            }
            sym.swap(next);
        }
        for (auto& s : sym) {
            auto it = vocab_.find(s);
            if (it != vocab_.end()) ids.push_back(it->second);
        }
    }

    // --- minimal UTF-8 ---
    static std::string utf8_encode(uint32_t cp) {
        std::string s;
        if (cp < 0x80) s.push_back((char)cp);
        else if (cp < 0x800) { s.push_back((char)(0xC0 | (cp >> 6))); s.push_back((char)(0x80 | (cp & 0x3F))); }
        else if (cp < 0x10000) { s.push_back((char)(0xE0 | (cp >> 12))); s.push_back((char)(0x80 | ((cp >> 6) & 0x3F))); s.push_back((char)(0x80 | (cp & 0x3F))); }
        else { s.push_back((char)(0xF0 | (cp >> 18))); s.push_back((char)(0x80 | ((cp >> 12) & 0x3F))); s.push_back((char)(0x80 | ((cp >> 6) & 0x3F))); s.push_back((char)(0x80 | (cp & 0x3F))); }
        return s;
    }
    static uint32_t utf8_next(const std::string& s, size_t& p) {
        unsigned char c = s[p];
        if (c < 0x80) { p += 1; return c; }
        if ((c >> 5) == 0x6) { uint32_t cp = ((c & 0x1F) << 6) | (s[p+1] & 0x3F); p += 2; return cp; }
        if ((c >> 4) == 0xE) { uint32_t cp = ((c & 0x0F) << 12) | ((s[p+1] & 0x3F) << 6) | (s[p+2] & 0x3F); p += 3; return cp; }
        uint32_t cp = ((c & 0x07) << 18) | ((s[p+1] & 0x3F) << 12) | ((s[p+2] & 0x3F) << 6) | (s[p+3] & 0x3F); p += 4; return cp;
    }
};

}  // namespace io
}  // namespace cppgrad
