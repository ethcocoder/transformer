#include "aurelis/lens/lens_model.hpp"
#include "aurelis/lens/bpe.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <numeric>

#include <filesystem>
namespace fs = std::filesystem;

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

void append_text_file(const std::string& file_path,
                      std::vector<std::string>& lines) {
    std::ifstream input(file_path.c_str());
    if (!input) return;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) lines.push_back(line);
    }
}

void append_text_files_from_directory(
    const std::string& directory_path,
    std::vector<std::string>& lines) {
    if (!fs::exists(directory_path) || !fs::is_directory(directory_path))
        return;
    for (const auto& entry : fs::directory_iterator(directory_path)) {
        if (entry.is_directory()) {
            append_text_files_from_directory(entry.path().string(), lines);
        } else if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            for (auto& c : ext)
                c = static_cast<char>(
                    std::tolower(static_cast<unsigned char>(c)));
            if (ext == ".txt" || ext == ".md") {
                append_text_file(entry.path().string(), lines);
            }
        }
    }
}

std::vector<std::string> load_dataset(const std::string& path) {
    std::vector<std::string> lines;
    if (!fs::exists(path)) {
        throw std::runtime_error("Dataset not found: " + path);
    }
    if (fs::is_regular_file(path)) {
        append_text_file(path, lines);
    } else if (fs::is_directory(path)) {
        append_text_files_from_directory(path, lines);
    }
    if (lines.empty()) {
        throw std::runtime_error("No text found in: " + path);
    }
    return lines;
}

float compute_accuracy(LensModel& model, const std::vector<int>& tokens,
                       int vocab_size) {
    if (tokens.size() < 2) return 0.0f;
    const auto result =
        model.forward(tokens.data(), static_cast<int>(tokens.size()));
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
        if (best_id == tokens[i + 1]) correct += 1.0f;
    }
    return correct / static_cast<float>(steps);
}

} // namespace

int main(int argc, char** argv) {
    std::string dataset_path = "data";
    std::string tokenizer_path = "";
    int epochs = 20;
    int hidden_dim = 0;
    int embed_dim = 0;
    int ff_dim = 0;
    int layers = 1;
    int vocab_size_target = 512;
    float lr = 0.01f;
    float lambda_aux = 0.01f;
    float lambda_stab = 0.01f;
    int batch_size = 32; // Default batch size

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc)
            dataset_path = argv[++i];
        else if (arg == "--tokenizer" && i + 1 < argc)
            tokenizer_path = argv[++i];
        else if (arg == "--epochs" && i + 1 < argc)
            epochs = parse_int_arg(argv[++i], epochs);
        else if (arg == "--hidden" && i + 1 < argc)
            hidden_dim = parse_int_arg(argv[++i], hidden_dim);
        else if (arg == "--embed" && i + 1 < argc)
            embed_dim = parse_int_arg(argv[++i], embed_dim);
        else if (arg == "--ff" && i + 1 < argc)
            ff_dim = parse_int_arg(argv[++i], ff_dim);
        else if (arg == "--layers" && i + 1 < argc)
            layers = parse_int_arg(argv[++i], layers);
        else if (arg == "--lr" && i + 1 < argc)
            lr = parse_float_arg(argv[++i], lr);
        else if (arg == "--vocab-size" && i + 1 < argc)
            vocab_size_target = parse_int_arg(argv[++i], vocab_size_target);
        else if (arg == "--lambda-aux" && i + 1 < argc)
            lambda_aux = parse_float_arg(argv[++i], lambda_aux);
        else if (arg == "--lambda-stab" && i + 1 < argc)
            lambda_stab = parse_float_arg(argv[++i], lambda_stab);
        else if (arg == "--batch-size" && i + 1 < argc)
            batch_size = parse_int_arg(argv[++i], batch_size);
    }

    /* Load data */
    const auto raw_lines = load_dataset(dataset_path);

    /* Train or load BPE tokenizer */
    BPETokenizer tokenizer;
    if (!tokenizer_path.empty() && fs::exists(tokenizer_path + ".vocab")) {
        tokenizer.load(tokenizer_path);
        printf("[BPE] Loaded tokenizer from %s (vocab=%d)\n",
               tokenizer_path.c_str(), tokenizer.vocab_size());
    } else {
        printf("[BPE] Training tokenizer on %zu lines, target vocab=%d\n",
               raw_lines.size(), vocab_size_target);
        tokenizer.train(raw_lines, vocab_size_target);
        if (!tokenizer_path.empty()) {
            tokenizer.save(tokenizer_path);
            printf("[BPE] Saved tokenizer to %s\n", tokenizer_path.c_str());
        }
    }

    const int vocab_size = tokenizer.vocab_size();

    /* Tokenize dataset */
    std::vector<DatasetRecord> dataset;
    dataset.reserve(raw_lines.size());
    for (const auto& line : raw_lines) {
        DatasetRecord rec;
        rec.text = line;
        rec.tokens = tokenizer.encode(line);
        if (!rec.tokens.empty()) {
            dataset.push_back(std::move(rec));
        }
    }

    const int split =
        static_cast<int>(dataset.size() * 8 / 10);
    const int resolved_hidden =
        hidden_dim > 0 ? hidden_dim
                       : std::max(32, std::min(128, vocab_size * 2));
    const int resolved_embed =
        embed_dim > 0 ? embed_dim
                      : std::max(16, std::min(64, resolved_hidden / 2));
    const int resolved_ff =
        ff_dim > 0 ? ff_dim : std::max(64, resolved_hidden * 4);

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

    printf("Aurelis LENS BPE training demo\n");
    printf("dataset=%s lines=%zu vocab=%d hidden=%d embed=%d ff=%d "
           "layers=%d lr=%.4f batch=%d train=%d eval=%d\n",
           dataset_path.c_str(), dataset.size(), vocab_size, cfg.D,
           cfg.d_model, cfg.d_ff, cfg.num_layers, cfg.lr, batch_size, split,
           static_cast<int>(dataset.size()) - split);

    float best_val_acc = 0.0f;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float train_loss = 0.0f;
        float train_acc = 0.0f;
        int train_steps = 0;

        // Shuffle training data
        std::vector<int> indices(split);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_shuffle(indices.begin(), indices.end());

        for (int i = 0; i < split; i += batch_size) {
            int current_batch_size = std::min(batch_size, split - i);
            
            if (i % 100 == 0 || i + current_batch_size >= split) {
                float progress = (float)(i + current_batch_size) / split * 100.0f;
                printf("\r[TRAIN] Epoch %d: %.1f%% (%d/%d)", epoch, progress, i + current_batch_size, split);
                fflush(stdout);
            }

            // Accumulate gradients over batch (simple version: sequential updates)
            // Since the model is currently optimized for single sequences, we'll keep sequential updates
            // but the structure is ready for true batching if forward/backward are updated.
            for (int b = 0; b < current_batch_size; ++b) {
                const auto& rec = dataset[indices[i + b]];
                const auto step = model.train_step(
                    rec.tokens.data(), static_cast<int>(rec.tokens.size()));
                train_loss += step.loss_total;
                
                // Optimized: Use logits from train_step instead of re-running forward
                float correct = 0.0f;
                const int steps = static_cast<int>(rec.tokens.size()) - 1;
                if (steps > 0) {
                    for (int t = 0; t < steps; ++t) {
                        const int base = t * vocab_size;
                        int best_id = 0;
                        float best_score = step.logits[base];
                        for (int j = 1; j < vocab_size; ++j) {
                            if (step.logits[base + j] > best_score) {
                                best_score = step.logits[base + j];
                                best_id = j;
                            }
                        }
                        if (best_id == rec.tokens[t + 1]) correct += 1.0f;
                    }
                    train_acc += correct / static_cast<float>(steps);
                }
                ++train_steps;
            }
        }
        printf("\n");

        float val_acc = 0.0f;
        int val_steps = 0;
        for (int i = split; i < static_cast<int>(dataset.size()); ++i) {
            const auto& rec = dataset[i];
            val_acc += compute_accuracy(model, rec.tokens, vocab_size);
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

        printf("epoch %2d  loss=%.4f  acc=%.3f  val_acc=%.3f\n", epoch,
               train_loss, train_acc, val_acc);
    }

    printf("best_val_acc=%.3f\n", best_val_acc);

    /* Print a generation sample */
    printf("\n--- Generation sample ---\n");
    if (!dataset.empty()) {
        const auto& seed = dataset[0];
        std::vector<int> gen_tokens = seed.tokens;
        // Keep at most 5 tokens from seed to start generation
        if (gen_tokens.size() > 5) gen_tokens.resize(5);
        
        printf("seed: %s\n", tokenizer.decode(gen_tokens).c_str());
        
        for (int t = 0; t < 20; ++t) {
            if (gen_tokens.size() < 2) {
                if (seed.tokens.size() >= 2) {
                    gen_tokens.push_back(seed.tokens[1]);
                } else {
                    gen_tokens.push_back(0); // Fallback
                }
                continue;
            }

            auto fwd = model.forward(gen_tokens.data(),
                                     static_cast<int>(gen_tokens.size()));
            
            if (fwd.logits.empty()) break;

            const int last = static_cast<int>(gen_tokens.size()) - 1;
            int best_id = 0;
            float best_score = fwd.logits[last * vocab_size];
            for (int v = 1; v < vocab_size; ++v) {
                if (fwd.logits[last * vocab_size + v] > best_score) {
                    best_score = fwd.logits[last * vocab_size + v];
                    best_id = v;
                }
            }
            gen_tokens.push_back(best_id);
            
            if (gen_tokens.size() > 50) break;
        }
        printf("gen:  %s\n", tokenizer.decode(gen_tokens).c_str());
    }

    return 0;
}
