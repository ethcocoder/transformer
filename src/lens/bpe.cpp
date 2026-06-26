#include "aurelis/lens/bpe.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <unordered_map>

namespace aurelis::lens {

namespace {

/* Count frequencies of adjacent pairs in a list of int-encoded words */
using PairCount = std::map<std::pair<int, int>, int64_t>;

struct Word {
    std::vector<int> tokens;
    int64_t count = 0;
};

void add_pair_count(PairCount& counts, int left, int right, int64_t inc) {
    if (inc == 0) return;
    auto key = std::make_pair(left, right);
    counts[key] += inc;
    if (counts[key] <= 0) counts.erase(key);
}

/* Replace all occurrences of a pair (l,r) in a sequence with new_id */
void replace_pair(std::vector<int>& seq, int left, int right, int new_id) {
    std::vector<int> result;
    result.reserve(seq.size());
    size_t i = 0;
    while (i < seq.size()) {
        if (i + 1 < seq.size() && seq[i] == left && seq[i + 1] == right) {
            result.push_back(new_id);
            i += 2;
        } else {
            result.push_back(seq[i]);
            i++;
        }
    }
    seq.swap(result);
}

} // anonymous namespace

int BPETokenizer::add_token(const std::string& token) {
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) return it->second;
    int id = static_cast<int>(id_to_token_.size());
    id_to_token_.push_back(token);
    token_to_id_[token] = id;
    return id;
}

void BPETokenizer::build_vocab() {
    token_to_id_.clear();
    for (int i = 0; i < static_cast<int>(id_to_token_.size()); ++i) {
        token_to_id_[id_to_token_[i]] = i;
    }
}

/* Convert UTF-8 text to byte-level IDs (0-255) */
std::vector<int> BPETokenizer::utf8_to_bytes(const std::string& text) {
    std::vector<int> bytes;
    bytes.reserve(text.size());
    for (unsigned char c : text) {
        bytes.push_back(static_cast<int>(c));
    }
    return bytes;
}

/* Convert byte-level IDs back to UTF-8 text */
std::string BPETokenizer::bytes_to_utf8(const std::vector<int>& bytes) {
    std::string result;
    result.reserve(bytes.size());
    for (int b : bytes) {
        result.push_back(static_cast<unsigned char>(b & 0xFF));
    }
    return result;
}

std::string BPETokenizer::token_to_printable(const std::string& token) {
    std::string result;
    for (unsigned char c : token) {
        if (c < 32 || c > 126) {
            result += "\\x" + std::to_string(c);
        } else {
            result += c;
        }
    }
    return result;
}

/* Core BPE training */
void BPETokenizer::train(const std::vector<std::string>& texts,
                         int vocab_size) {
    if (vocab_size < 256) {
        throw std::invalid_argument(
            "BPETokenizer: vocab_size must be >= 256");
    }

    /* Initialize with byte-level vocabulary (0-255) */
    id_to_token_.clear();
    id_to_token_.reserve(static_cast<size_t>(vocab_size));
    for (int i = 0; i < 256; ++i) {
        std::string s(1, static_cast<unsigned char>(i));
        add_token(s);
    }

    /* Add special tokens */
    add_token("<pad>");
    add_token("<s>");
    add_token("</s>");
    add_token("<unk>");

    /* Convert corpus to lists of byte IDs, weighted by word frequency */
    std::unordered_map<std::string, int64_t> word_counts;
    for (const auto& text : texts) {
        std::istringstream stream(text);
        std::string word;
        while (stream >> word) {
            word_counts[word]++;
        }
    }

    if (word_counts.empty()) {
        /* Fallback: treat entire text as one word */
        for (const auto& text : texts) {
            word_counts[text]++;
        }
    }

    std::vector<Word> corpus;
    corpus.reserve(word_counts.size());
    for (const auto& wc : word_counts) {
        Word w;
        w.tokens = utf8_to_bytes(wc.first);
        w.count = wc.second;
        corpus.push_back(std::move(w));
    }

    /* If no merges needed */
    int num_merges = vocab_size - static_cast<int>(id_to_token_.size());
    if (num_merges <= 0) {
        build_vocab();
        return;
    }

    /* Iteratively learn merges */
    merges_.clear();
    merges_.reserve(static_cast<size_t>(num_merges));

    for (int merge_step = 0; merge_step < num_merges; ++merge_step) {
        /* Count all adjacent pairs across the corpus */
        PairCount pair_counts;
        for (const auto& word : corpus) {
            for (size_t i = 0; i + 1 < word.tokens.size(); ++i) {
                add_pair_count(pair_counts, word.tokens[i],
                               word.tokens[i + 1], word.count);
            }
        }

        if (pair_counts.empty()) {
            break; /* No more pairs to merge */
        }

        /* Find the most frequent pair */
        auto best = pair_counts.begin();
        for (auto it = pair_counts.begin(); it != pair_counts.end(); ++it) {
            if (it->second > best->second) {
                best = it;
            }
        }

        int left = best->first.first;
        int right = best->first.second;

        /* Create a new token for this merge */
        std::string merged_token = id_to_token_[left] + id_to_token_[right];
        int new_id = add_token(merged_token);

        BPEMerge merge;
        merge.left = left;
        merge.right = right;
        merge.new_id = new_id;
        merges_.push_back(merge);

        /* Apply merge to all words in the corpus */
        for (auto& word : corpus) {
            replace_pair(word.tokens, left, right, new_id);
        }

        if (static_cast<int>(id_to_token_.size()) >= vocab_size) {
            break;
        }
    }

    build_vocab();
}

/* Encode text: convert to bytes, then apply BPE merges greedily */
std::vector<int> BPETokenizer::encode(const std::string& text) const {
    if (id_to_token_.empty()) {
        /* Fallback: raw bytes */
        return utf8_to_bytes(text);
    }

    auto byte_ids = utf8_to_bytes(text);
    return bpe_encode(byte_ids);
}

/* Greedy BPE encoding: always apply the highest-priority (earliest learned) merge */
std::vector<int> BPETokenizer::bpe_encode(
    const std::vector<int>& byte_ids) const {
    std::vector<int> tokens = byte_ids;

    if (merges_.empty()) {
        return tokens;
    }

    /* Build a quick lookup: pair -> merge priority (index in merges_) */
    std::map<std::pair<int, int>, int> merge_priority;
    for (int i = 0; i < static_cast<int>(merges_.size()); ++i) {
        merge_priority[std::make_pair(merges_[i].left, merges_[i].right)] = i;
    }

    bool changed = true;
    while (changed) {
        changed = false;
        /* Find the best merge to apply (lowest priority index = earliest merge) */
        int best_priority = std::numeric_limits<int>::max();
        int best_pos = -1;
        int best_left = 0, best_right = 0, best_new_id = 0;

        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            auto it = merge_priority.find(
                std::make_pair(tokens[i], tokens[i + 1]));
            if (it != merge_priority.end() && it->second < best_priority) {
                best_priority = it->second;
                best_pos = static_cast<int>(i);
                best_left = tokens[i];
                best_right = tokens[i + 1];
                best_new_id = merges_[it->second].new_id;
            }
        }

        if (best_pos >= 0) {
            /* Apply the best merge */
            std::vector<int> new_tokens;
            new_tokens.reserve(tokens.size());
            size_t i = 0;
            while (i < tokens.size()) {
                if (static_cast<int>(i) == best_pos && i + 1 < tokens.size()) {
                    new_tokens.push_back(best_new_id);
                    i += 2;
                } else {
                    new_tokens.push_back(tokens[i]);
                    i++;
                }
            }
            tokens.swap(new_tokens);
            changed = true;
        }
    }

    return tokens;
}

/* Decode token IDs back to text */
std::string BPETokenizer::decode(const std::vector<int>& tokens) const {
    std::vector<int> byte_seq;
    byte_seq.reserve(tokens.size());

    for (int id : tokens) {
        if (id < 0 || id >= static_cast<int>(id_to_token_.size())) {
            continue; /* Skip invalid IDs */
        }
        const std::string& token = id_to_token_[id];
        /* Skip special tokens */
        if (token == "<pad>" || token == "<s>" || token == "</s>" ||
            token == "<unk>") {
            continue;
        }
        /* Convert token back to bytes */
        for (unsigned char c : token) {
            byte_seq.push_back(static_cast<int>(c));
        }
    }

    return bytes_to_utf8(byte_seq);
}

int BPETokenizer::special_token_id(const std::string& token) const {
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) return it->second;
    if (token == "<unk>") return 0; /* fallback: first byte */
    return 0; /* fallback */
}

bool BPETokenizer::save(const std::string& prefix) const {
    /* Save vocab */
    {
        std::ofstream f(prefix + ".vocab");
        if (!f) return false;
        for (const auto& s : id_to_token_) {
            /* Escape newlines in token */
            std::string escaped = s;
            for (size_t p = escaped.find('\n'); p != std::string::npos;
                 p = escaped.find('\n', p)) {
                escaped.replace(p, 1, "\\n");
                p += 2;
            }
            f << escaped << "\n";
        }
    }

    /* Save merges */
    {
        std::ofstream f(prefix + ".merges");
        if (!f) return false;
        for (const auto& m : merges_) {
            f << m.left << " " << m.right << " " << m.new_id << "\n";
        }
    }

    return true;
}

bool BPETokenizer::load(const std::string& prefix) {
    id_to_token_.clear();
    token_to_id_.clear();
    merges_.clear();

    /* Load vocab */
    {
        std::ifstream f(prefix + ".vocab");
        if (!f) return false;
        std::string line;
        while (std::getline(f, line)) {
            /* Unescape */
            std::string token = line;
            for (size_t p = token.find("\\n"); p != std::string::npos;
                 p = token.find("\\n", p)) {
                token.replace(p, 2, "\n");
                p += 1;
            }
            add_token(token);
        }
    }

    /* Load merges */
    {
        std::ifstream f(prefix + ".merges");
        if (!f) return false;
        BPEMerge m;
        while (f >> m.left >> m.right >> m.new_id) {
            merges_.push_back(m);
        }
    }

    build_vocab();
    return true;
}

} // namespace aurelis::lens
