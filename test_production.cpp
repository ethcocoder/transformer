#include <iostream>
#include "aurelis/config.hpp"
#include "aurelis/logging.hpp"
#include "aurelis/profiler.hpp"
#include "aurelis/tensor.hpp"

int main() {
    std::cout << "Testing production features..." << std::endl;
    
    // Test logger
    aurelis::Logger::instance().set_log_level(aurelis::LogLevel::DEBUG);
    LOG_INFO("Test log message");
    LOG_DEBUG("Debug test");
    
    // Test profiler
    {
        aurelis::ScopedProfiler sp("test");
        // Do some work
        for (int i = 0; i < 1000000; i++) {}
    }
    aurelis::Profiler::instance().print_report();
    
    // Test config
    aurelis::AurelisConfig cfg;
    cfg.lens.vocab_size = 32;
    cfg.save("test_config.json");
    auto loaded_cfg = aurelis::AurelisConfig::load("test_config.json");
    std::cout << "Loaded vocab size: " << loaded_cfg.lens.vocab_size << std::endl;
    
    // Test tensor save/load
    aurelis::Tensor t({2, 3});
    t.fill(1.0f);
    t.save("test_tensor.bin");
    auto loaded_t = aurelis::Tensor::load("test_tensor.bin");
    std::cout << "Loaded tensor numel: " << loaded_t.numel() << std::endl;
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
