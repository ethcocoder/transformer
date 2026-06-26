#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

namespace aurelis::lens {

struct BPEMerge {
    int left;
    int right;
    int new_id;
};

class BPETokenizer {
public:
    BPETokenizer() = default;

    /* Train BPE from a list of texts up to vocab_size tokens.
     * vocab_size must be >= 256 (byte-level base). */
    void train(const std::vector<std::string>& texts, int vocab_size);

    /* Encode text into token IDs. Returns unknown tokens as their byte IDs. */
    std::vector<int> encode(const std::string& text) const;

    /* Decode token IDs back to text. */
    std::string decode(const std::vector<int>& tokens) const;

    /* Save/load vocabulary and merges. */
    bool save(const std::string& prefix) const;
    bool load(const std::string& prefix);

    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }
    int special_token_id(const std::string& token) const;
    int pad_id() const { return special_token_id("<pad>"); }
    int bos_id() const { return special_token_id("<s>"); }
    int eos_id() const { return special_token_id("</s>"); }
    int unk_id() const { return special_token_id("<unk>"); }

    const std::vector<std::string>& id_to_token() const { return id_to_token_; }

private:
    std::vector<std::string> id_to_token_;       /* id -> token string */
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<BPEMerge> merges_;               /* ordered list of learned merges */

    /* Internal: split text into subword units using current merges */
    std::vector<int> bpe_encode(const std::vector<int>& byte_ids) const;

    int add_token(const std::string& token);
    void build_vocab();

    static std::vector<int> utf8_to_bytes(const std::string& text);
    static std::string bytes_to_utf8(const std::vector<int>& bytes);
    static std::string token_to_printable(const std::string& token);
};

} // namespace aurelis::lens
