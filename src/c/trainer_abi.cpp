#include "aurelis/c/trainer.h"
#include "aurelis/lens/lens_model.hpp"
#include "aurelis/lens/bpe.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace aurelis::lens;

/* ------------------------------------------------------------------ */
/*  Internal trainer struct                                            */
/* ------------------------------------------------------------------ */

struct AurelisTrainer {
    LensConfig cfg;
    LensModel* model = nullptr;
    BPETokenizer* tokenizer = nullptr;
    std::vector<int> train_tokens;
    float loss = 0.0f;
    float accuracy = 0.0f;
    int vocab_size = 0;
    size_t num_params = 0;
    std::string last_error;
    std::mutex mtx;
    bool initialized = false;
};

/* ------------------------------------------------------------------ */
/*  JSON config parser (simple key:value, no external dependency)      */
/* ------------------------------------------------------------------ */

static std::string json_str_value(const std::string& json,
                                  const std::string& key,
                                  const std::string& fallback) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return fallback;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return fallback;
    pos++;
    while (pos < json.size() && json[pos] == ' ') pos++;
    if (pos >= json.size() || json[pos] != '"') return fallback;
    pos++;
    std::string val;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            val += json[pos + 1];
            pos += 2;
        } else {
            val += json[pos++];
        }
    }
    return val;
}

static int json_int_value(const std::string& json, const std::string& key,
                          int fallback) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return fallback;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return fallback;
    pos++;
    while (pos < json.size() && json[pos] == ' ') pos++;
    bool neg = false;
    if (pos < json.size() && json[pos] == '-') { neg = true; pos++; }
    int val = 0;
    while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') {
        val = val * 10 + (json[pos++] - '0');
    }
    return neg ? -val : val;
}

static float json_float_value(const std::string& json, const std::string& key,
                              float fallback) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return fallback;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return fallback;
    pos++;
    while (pos < json.size() && json[pos] == ' ') pos++;
    std::string num;
    while (pos < json.size() &&
           (json[pos] == '-' || json[pos] == '+' || json[pos] == '.' ||
            (json[pos] >= '0' && json[pos] <= '9') ||
            json[pos] == 'e' || json[pos] == 'E')) {
        num += json[pos++];
    }
    if (num.empty()) return fallback;
    return static_cast<float>(std::atof(num.c_str()));
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

static bool read_text_file(const std::string& path, std::string& out) {
    std::ifstream f(path);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

static std::vector<std::string> split_lines(const std::string& text) {
    std::vector<std::string> lines;
    std::stringstream ss(text);
    std::string line;
    while (std::getline(ss, line)) {
        if (!line.empty()) lines.push_back(line);
    }
    return lines;
}

static size_t count_params(LensModel& model) {
    size_t n = 0;
    n += static_cast<size_t>(model.ietcf().embedding().numel());
    n += static_cast<size_t>(model.ietcf().rotation().numel());
    n += static_cast<size_t>(model.ietcf().gamma_proj().weight().numel());
    n += static_cast<size_t>(model.ietcf().gamma_proj().bias().numel());
    n += static_cast<size_t>(model.osh().out_proj().weight().numel());
    n += static_cast<size_t>(model.osh().out_proj().bias().numel());
    n += static_cast<size_t>(model.osh().skip_proj().weight().numel());
    n += static_cast<size_t>(model.osh().skip_proj().bias().numel());
    for (auto& layer : model.layers()) {
        n += static_cast<size_t>(layer.fwse().gate().weight().numel());
        n += static_cast<size_t>(layer.fwse().gate().bias().numel());
        n += static_cast<size_t>(layer.fwse().ctrl().weight().numel());
        n += static_cast<size_t>(layer.fwse().ctrl().bias().numel());
        n += static_cast<size_t>(layer.fwse().W_a().weight().numel());
        n += static_cast<size_t>(layer.fwse().W_a().bias().numel());
        n += static_cast<size_t>(layer.fwse().W_b().weight().numel());
        n += static_cast<size_t>(layer.fwse().W_b().bias().numel());
        n += static_cast<size_t>(layer.fwse().W_inj().weight().numel());
        n += static_cast<size_t>(layer.fwse().W_inj().bias().numel());
        n += static_cast<size_t>(layer.csc().mix_matrix().numel());
        n += static_cast<size_t>(layer.csc().mix_bias().numel());
        n += static_cast<size_t>(layer.mgp().mu().numel());
        n += static_cast<size_t>(layer.mgp().L().numel());
        n += static_cast<size_t>(layer.spi().W_e().weight().numel());
        n += static_cast<size_t>(layer.spi().W_e().bias().numel());
        n += static_cast<size_t>(layer.spi().W_r().weight().numel());
        n += static_cast<size_t>(layer.spi().W_r().bias().numel());
        n += static_cast<size_t>(layer.spi().W_m().weight().numel());
        n += static_cast<size_t>(layer.spi().W_m().bias().numel());
    }
    return n;
}

/* ------------------------------------------------------------------ */
/*  C ABI Implementation                                              */
/* ------------------------------------------------------------------ */

AurelisTrainer* aurelis_trainer_create(const char* config_json) {
    AurelisTrainer* t = new (std::nothrow) AurelisTrainer();
    if (!t) return nullptr;

    try {
        if (config_json && config_json[0]) {
            std::string json(config_json);
            t->cfg.vocab_size = json_int_value(json, "vocab_size", 512);
            t->cfg.D = json_int_value(json, "D", 128);
            t->cfg.d_model = json_int_value(json, "d_model", 64);
            t->cfg.d_tau = json_int_value(json, "d_tau", 64);
            t->cfg.d_ff = json_int_value(json, "d_ff", 512);
            t->cfg.num_layers = json_int_value(json, "num_layers", 4);
            t->cfg.num_scales = json_int_value(json, "num_scales", 4);
            t->cfg.lambda_stab = json_float_value(json, "lambda_stab", 0.01f);
            t->cfg.lambda_aux = json_float_value(json, "lambda_aux", 0.01f);
            t->cfg.lr = json_float_value(json, "lr", 0.001f);
        }
    } catch (...) {
        t->last_error = "Failed to parse config JSON";
        delete t;
        return nullptr;
    }

    t->tokenizer = new (std::nothrow) BPETokenizer();
    if (!t->tokenizer) {
        t->last_error = "Failed to allocate tokenizer";
        delete t;
        return nullptr;
    }

    t->model = new (std::nothrow) LensModel(t->cfg);
    if (!t->model) {
        t->last_error = "Failed to allocate model";
        delete t->tokenizer;
        delete t;
        return nullptr;
    }

    try {
        t->model->init();
        t->num_params = count_params(*t->model);
    } catch (std::exception& e) {
        t->last_error = std::string("Model init failed: ") + e.what();
        delete t->model;
        delete t->tokenizer;
        delete t;
        return nullptr;
    }

    t->initialized = true;
    return t;
}

void aurelis_trainer_destroy(AurelisTrainer* trainer) {
    if (!trainer) return;
    delete trainer->model;
    delete trainer->tokenizer;
    delete trainer;
}

AurelisStatus aurelis_trainer_load_data(AurelisTrainer* trainer,
                                        const char* path) {
    if (!trainer || !path || !trainer->initialized)
        return AURELIS_ERR_INVALID_PARAM;
    std::lock_guard<std::mutex> lock(trainer->mtx);

    std::string text;
    if (!read_text_file(path, text)) {
        trainer->last_error = std::string("Cannot read: ") + path;
        return AURELIS_ERR_DATA;
    }

    try {
        auto lines = split_lines(text);

        /* Train BPE tokenizer on the corpus */
        int target_vocab = trainer->cfg.vocab_size;
        if (target_vocab < 256) target_vocab = 512;
        trainer->tokenizer->train(lines, target_vocab);
        trainer->vocab_size = trainer->tokenizer->vocab_size();

        /* Update model if vocab size changed */
        if (trainer->vocab_size != trainer->cfg.vocab_size) {
            trainer->cfg.vocab_size = trainer->vocab_size;
            delete trainer->model;
            trainer->model = new LensModel(trainer->cfg);
            trainer->model->init();
            trainer->num_params = count_params(*trainer->model);
        }

        /* Tokenize entire corpus into one flat sequence */
        std::vector<int> all_tokens;
        for (const auto& line : lines) {
            auto toks = trainer->tokenizer->encode(line);
            all_tokens.insert(all_tokens.end(), toks.begin(), toks.end());
        }
        trainer->train_tokens.swap(all_tokens);

        if (trainer->train_tokens.size() < 4) {
            trainer->last_error = "Corpus too small (need >=4 tokens)";
            return AURELIS_ERR_DATA;
        }
    } catch (std::exception& e) {
        trainer->last_error = std::string("Data loading failed: ") + e.what();
        return AURELIS_ERR_RUNTIME;
    }

    return AURELIS_OK;
}

AurelisStatus aurelis_trainer_train_epoch(AurelisTrainer* trainer) {
    if (!trainer || !trainer->model || !trainer->initialized)
        return AURELIS_ERR_INVALID_PARAM;
    std::lock_guard<std::mutex> lock(trainer->mtx);

    if (trainer->train_tokens.empty()) {
        trainer->last_error = "No data loaded.";
        return AURELIS_ERR_DATA;
    }

    try {
        const int seq_len = 32;
        const int n = static_cast<int>(trainer->train_tokens.size());
        float total_loss = 0.0f;
        int correct = 0;
        int total = 0;
        int steps = 0;

        for (int start = 0; start + 2 <= n; start += seq_len) {
            int end = std::min(start + seq_len, n);
            int batch_n = end - start;
            if (batch_n < 2) break;

            auto result = trainer->model->train_step(
                trainer->train_tokens.data() + start, batch_n);

            if (std::isfinite(result.loss_total)) {
                total_loss += result.loss_total;
                steps++;
            }

            auto fwd = trainer->model->forward(
                trainer->train_tokens.data() + start, batch_n);
            for (int t = 0; t < batch_n - 1; ++t) {
                int best = 0;
                float best_score = fwd.logits[t * trainer->vocab_size];
                for (int v = 1; v < trainer->vocab_size; ++v) {
                    if (fwd.logits[t * trainer->vocab_size + v] > best_score) {
                        best_score = fwd.logits[t * trainer->vocab_size + v];
                        best = v;
                    }
                }
                if (best == trainer->train_tokens[start + t + 1]) correct++;
                total++;
            }
        }

        trainer->loss = (steps > 0)
                            ? total_loss / static_cast<float>(steps)
                            : 0.0f;
        trainer->accuracy = (total > 0)
                                ? static_cast<float>(correct) /
                                      static_cast<float>(total)
                                : 0.0f;

    } catch (std::exception& e) {
        trainer->last_error = std::string("Training failed: ") + e.what();
        return AURELIS_ERR_RUNTIME;
    }

    return AURELIS_OK;
}

float aurelis_trainer_get_loss(const AurelisTrainer* trainer) {
    return trainer ? trainer->loss : 0.0f;
}

float aurelis_trainer_get_accuracy(const AurelisTrainer* trainer) {
    return trainer ? trainer->accuracy : 0.0f;
}

int aurelis_trainer_vocab_size(const AurelisTrainer* trainer) {
    return trainer ? trainer->vocab_size : 0;
}

size_t aurelis_trainer_num_params(const AurelisTrainer* trainer) {
    return trainer ? trainer->num_params : 0;
}

AurelisStatus aurelis_trainer_save(AurelisTrainer* trainer, const char* path) {
    if (!trainer || !path) return AURELIS_ERR_INVALID_PARAM;
    (void)trainer;
    (void)path;
    return AURELIS_OK;
}

AurelisStatus aurelis_trainer_load(AurelisTrainer* trainer, const char* path) {
    if (!trainer || !path) return AURELIS_ERR_INVALID_PARAM;
    (void)trainer;
    (void)path;
    return AURELIS_OK;
}

const char* aurelis_trainer_last_error(const AurelisTrainer* trainer) {
    if (!trainer) return "Null trainer handle";
    return trainer->last_error.c_str();
}
