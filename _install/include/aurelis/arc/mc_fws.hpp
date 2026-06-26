#pragma once

#include <aurelis/arc/config.hpp>
#include <aurelis/state_partition.hpp>
#include <aurelis/lens/mssp.hpp>
#include <vector>

namespace aurelis::arc {

// Multiscale Fractal Wave Scanner (MC-FWS)
class MCFWS {
public:
    explicit MCFWS(const ArcConfig& cfg);

    void init();

    // Compute (a^r, b^r) for r channel
    void compute_fw_params(const float* c, const float* gamma, const float* xi_eff, float epsilon_eff, float alpha_create,
                          float* a_r, float* b_r, int n) const;

private:
    ArcConfig cfg;
    StatePartition part;
    aurelis::lens::MsspLayout mssp_layout;
};

} // namespace aurelis::arc
