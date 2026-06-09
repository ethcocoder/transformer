#pragma once

#include <aurelis/arc/config.hpp>
#include <aurelis/tensor.hpp>
#include <aurelis/state_partition.hpp>
#include <vector>

namespace aurelis::arc {

// Reasoning Scan Engine (RSE)
class RSE {
public:
    explicit RSE(const ArcConfig& cfg);

    void init();

    // Scan r using (a_r, b_r)
    void forward_scan(const float* a_r, const float* b_r, int n, float* r) const;

private:
    ArcConfig cfg;
};

} // namespace aurelis::arc
