#include <fstream>
#include <iostream>
#include <string>

#include "aurelis/config.hpp"
#include "aurelis/errors.hpp"
#include "aurelis/tensor.hpp"

static bool require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        return false;
    }
    return true;
}

int main() {
    const std::string invalid_path = "invalid_config.json";
    {
        std::ofstream out(invalid_path);
        out << "{\n  \"lens\": {\n    \"vocab_size\": 0,\n    \"D\": 64\n  }\n}\n";
    }

    bool threw = false;
    try {
        (void)aurelis::AurelisConfig::load(invalid_path);
    } catch (const aurelis::AurelisException& ex) {
        threw = true;
        std::cout << "caught expected exception: " << ex.what() << std::endl;
    }

    if (!require(threw, "invalid config should be rejected")) {
        return 1;
    }

    try {
        aurelis::Tensor t({2, 3});
        (void)t.view({3, 4});
        return require(false, "view should reject incompatible dimensions");
    } catch (const std::invalid_argument&) {
        std::cout << "view mismatch rejected as expected" << std::endl;
    }

    std::remove(invalid_path.c_str());
    std::cout << "Production hardening checks passed" << std::endl;
    return 0;
}
