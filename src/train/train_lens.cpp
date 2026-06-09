#include "aurelis/lens/lens_model.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

#include <windows.h>

using namespace aurelis::lens;

namespace {

int parse_int_arg(const char* value, int fallback) {
    if (!value) return fallback;
    return std::atoi(value);
}

float parse_float_arg(const char* value, float fallback) {
    if (!value) return fallback;
    return static_cast<float>(std::atof(value));
}

struct DatasetRecord {
    std::string text;
    std::vector<int> tokens;
};

bool path_exists(const std::string& path) {
    struct _stat info;
    return _stat(path.c_str(), &info) == 0;
}

bool path_is_directory(const std::string& path) {
    struct _stat info;
    if (_stat(path.c_str(), &info) != 0) {
        return false;
    }
    return (info.st_mode & _S_IFDIR) != 0;
}

bool path_is_regular_file(const std::string& path) {
    struct _stat info;
    if (_stat(path.c_str(), &info) != 0) {
        return false;
    }
    return (info.st_mode & _S_IFREG) != 0;
}

std::string to_lower_copy(const std::string& value) {
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return lower;
}

void append_text_file(const std::string& file_path, std::vector<std::string>& lines) {
    std::ifstream input(file_path.c_str());
    if (!input) {
        return;
    }

    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
}

void append_text_files_from_directory(const std::string& directory_path,
                                      std::vector<std::string>& lines) {
    const std::string search = directory_path + "\\*";
    WIN32_FIND_DATAA find_data{};
    const HANDLE handle = FindFirstFileA(search.c_str(), &find_data);
    if (handle == INVALID_HANDLE_VALUE) {
        return;
    }

    do {
        const std::string entry_name(find_data.cFileName);
        if (entry_name == "." || entry_name == "..") {
            continue;
        }

        const std::string full_path = directory_path + "\\" + entry_name;
        if (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            append_text_files_from_directory(full_path, lines);
            continue;
        }

        const std::string ext = to_lower_copy(entry_name.substr(entry_name.find_last_of('.')));
        if (ext != ".txt" && ext != ".md") {
            continue;
        }

        append_text_file(full_path, lines);
    } while (FindNextFileA(handle, &find_data));

    FindClose(handle);
}

std::vector<std::string> load_dataset_from_path(const std::string& path) {
    std::vector<std::string> lines;

    if (path_is_regular_file(path)) {
        append_text_file(path, lines);
    } else if (path_is_directory(path)) {
        append_text_files_from_directory(path, lines);
    } else {
        throw std::runtime_error("Dataset path does not exist: " + path);
    }

    if (lines.empty()) {
        throw std::runtime_error("Dataset path contains no readable text: " + path);
    }
    return lines;
}

std::unordered_map<char, int> build_vocabulary(const std::vector<std::string>& sentences) {
    std::unordered_map<char, int> vocab;
    int next_id = 0;
    for (const std::string& sentence : sentences) {
        for (char ch : sentence) {
            if (vocab.find(ch) == vocab.end()) {
                vocab[ch] = next_id++;
            }
        }
    }
    // Keep a simple, stable set of common characters for the tiny dataset.
    const std::string extra_chars = " abcdefghijklmnopqrstuvwxyz.";
    for (char ch : extra_chars) {
        if (vocab.find(ch) == vocab.end()) {
            vocab[ch] = next_id++;
        }
    }
    return vocab;
}

std::vector<int> tokenize(const std::string& text,
                          const std::unordered_map<char, int>& vocab) {
    std::vector<int> tokens;
    for (char ch : text) {
        const auto it = vocab.find(ch);
        if (it != vocab.end()) {
            tokens.push_back(it->second);
        }
    }
    return tokens;
}

float compute_accuracy(LensModel& model, const std::vector<int>& tokens,
                       int vocab_size) {
    if (tokens.size() < 2) {
        return 0.0f;
    }

    const auto result = model.forward(tokens.data(), static_cast<int>(tokens.size()));
    float correct = 0.0f;
    const int steps = static_cast<int>(tokens.size()) - 1;
    for (int i = 0; i < steps; ++i) {
        const int base = i * vocab_size;
        int best_id = 0;
        float best_score = result.logits[base];
        for (int j = 1; j < vocab_size; ++j) {
            if (result.logits[base + j] > best_score) {
                best_score = result.logits[base + j];
                best_id = j;
            }
        }
        if (best_id == tokens[i + 1]) {
            correct += 1.0f;
        }
    }
    return correct / static_cast<float>(steps);
}

}  // namespace

int main(int argc, char** argv) {
    std::string dataset_path = "data";
    int epochs = 20;
    int hidden_dim = 0;
    int embed_dim = 0;
    int ff_dim = 0;
    int layers = 1;
    float lr = 0.01f;
    float lambda_aux = 0.01f;
    float lambda_stab = 0.01f;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) dataset_path = argv[++i];
        else if (arg == "--epochs" && i + 1 < argc) epochs = parse_int_arg(argv[++i], epochs);
        else if (arg == "--hidden" && i + 1 < argc) hidden_dim = parse_int_arg(argv[++i], hidden_dim);
        else if (arg == "--embed" && i + 1 < argc) embed_dim = parse_int_arg(argv[++i], embed_dim);
        else if (arg == "--ff" && i + 1 < argc) ff_dim = parse_int_arg(argv[++i], ff_dim);
        else if (arg == "--layers" && i + 1 < argc) layers = parse_int_arg(argv[++i], layers);
        else if (arg == "--lr" && i + 1 < argc) lr = parse_float_arg(argv[++i], lr);
        else if (arg == "--lambda-aux" && i + 1 < argc) lambda_aux = parse_float_arg(argv[++i], lambda_aux);
        else if (arg == "--lambda-stab" && i + 1 < argc) lambda_stab = parse_float_arg(argv[++i], lambda_stab);
    }

    const std::vector<std::string> raw_sentences = load_dataset_from_path(dataset_path);
    const auto vocab = build_vocabulary(raw_sentences);

    std::vector<DatasetRecord> dataset;
    dataset.reserve(raw_sentences.size());
    for (const std::string& text : raw_sentences) {
        DatasetRecord record;
        record.text = text;
        record.tokens = tokenize(text, vocab);
        if (!record.tokens.empty()) {
            dataset.push_back(std::move(record));
        }
    }

    const int vocab_size = static_cast<int>(vocab.size());
    const int split = static_cast<int>(dataset.size() * 8 / 10);
    const int resolved_hidden = hidden_dim > 0 ? hidden_dim : std::max(16, std::min(64, vocab_size * 2));
    const int resolved_embed = embed_dim > 0 ? embed_dim : std::max(8, std::min(32, resolved_hidden / 2));
    const int resolved_ff = ff_dim > 0 ? ff_dim : std::max(32, resolved_hidden * 4);

    LensConfig cfg;
    cfg.vocab_size = vocab_size;
    cfg.D = resolved_hidden;
    cfg.d_model = resolved_embed;
    cfg.d_tau = resolved_embed;
    cfg.d_ff = resolved_ff;
    cfg.num_layers = layers;
    cfg.lr = lr;
    cfg.lambda_aux = lambda_aux;
    cfg.lambda_stab = lambda_stab;

    LensModel model(cfg);
    model.init();

    printf("Aurelis LENS small-language training demo\n");
    printf("dataset=%s sentences=%zu vocab_size=%d hidden=%d embed=%d ff=%d layers=%d lr=%.4f lambda_aux=%.4f lambda_stab=%.4f train=%d eval=%d\n",
           dataset_path.c_str(), dataset.size(), vocab_size, cfg.D, cfg.d_model, cfg.d_ff,
           cfg.num_layers, cfg.lr, cfg.lambda_aux, cfg.lambda_stab,
           split, static_cast<int>(dataset.size()) - split);

    float best_val_acc = 0.0f;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float train_loss = 0.0f;
        float train_acc = 0.0f;
        int train_steps = 0;

        for (int i = 0; i < split; ++i) {
            const auto& record = dataset[i];
            const auto step = model.train_step(record.tokens.data(),
                                              static_cast<int>(record.tokens.size()));
            train_loss += step.loss_total;
            train_acc += compute_accuracy(model, record.tokens, vocab_size);
            ++train_steps;
        }

        float val_acc = 0.0f;
        int val_steps = 0;
        for (int i = split; i < static_cast<int>(dataset.size()); ++i) {
            const auto& record = dataset[i];
            val_acc += compute_accuracy(model, record.tokens, vocab_size);
            ++val_steps;
        }

        if (val_steps > 0) {
            val_acc /= static_cast<float>(val_steps);
            best_val_acc = std::max(best_val_acc, val_acc);
        }

        if (train_steps > 0) {
            train_loss /= static_cast<float>(train_steps);
            train_acc /= static_cast<float>(train_steps);
        }

        printf("epoch %2d  train_loss=%.4f  train_acc=%.3f  val_acc=%.3f\n",
               epoch, train_loss, train_acc, val_acc);
    }

    printf("best_val_acc=%.3f\n", best_val_acc);
    return 0;
}
