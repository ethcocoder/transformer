#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "aurelis/config.hpp"
#include "aurelis/logging.hpp"
#include "aurelis/profiler.hpp"
#include "aurelis/tensor.hpp"

static bool require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        return false;
    }
    return true;
}

int main() {
    std::cout << "Testing production features..." << std::endl;

    aurelis::Logger::instance().set_log_level(aurelis::LogLevel::DEBUG);
    LOG_INFO("Test log message");
    LOG_DEBUG("Debug test");

    {
        aurelis::ScopedProfiler sp("production_smoke");
        volatile int sink = 0;
        for (int i = 0; i < 10000; ++i) {
            sink += i;
        }
        (void)sink;
    }
    aurelis::Profiler::instance().print_report();

    aurelis::AurelisConfig cfg;
    cfg.lens.vocab_size = 32;
    if (!cfg.save("test_config.json")) {
        std::cerr << "FAIL: config save failed" << std::endl;
        return 1;
    }

    std::ifstream config_stream("test_config.json", std::ios::binary);
    if (!require(config_stream.good(), "config file was not created")) {
        return 1;
    }

    auto loaded_cfg = aurelis::AurelisConfig::load("test_config.json");
    if (!require(loaded_cfg.lens.vocab_size == 32, "config round-trip failed")) {
        return 1;
    }

    aurelis::Tensor t({2, 3});
    t.fill(1.0f);
    if (!t.save("test_tensor.bin")) {
        std::cerr << "FAIL: tensor save failed" << std::endl;
        return 1;
    }
    std::ifstream tensor_stream("test_tensor.bin", std::ios::binary);
    if (!require(tensor_stream.good(), "tensor file was not created")) {
        return 1;
    }

    auto loaded_t = aurelis::Tensor::load("test_tensor.bin");
    if (!require(loaded_t.numel() == 6, "tensor load size mismatch")) {
        return 1;
    }
    if (!require(loaded_t.at(0) == 1.0f, "tensor load value mismatch")) {
        return 1;
    }

    std::remove("test_config.json");
    std::remove("test_tensor.bin");

    std::cout << "All production smoke tests passed!" << std::endl;
    return 0;
}
