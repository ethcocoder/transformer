#pragma once

#include <aurelis/arc/config.hpp>
#include <aurelis/lens/linear.hpp>
#include <aurelis/tensor.hpp>
#include <aurelis/state_partition.hpp>
#include <vector>

namespace aurelis::arc {

// Attractor Learning Engine (ALE)
class ALE {
public:
    explicit ALE(const ArcConfig& cfg);

    void init();

    // ASA: Attractor Selection Attention
    void select_attractor(const float* c, const float* e, const float* alpha, float* pi) const;

    // GBS: Get Basin Geometry
    void get_basin_precision(int m, float* Lm) const; // Lower triangular Cholesky factor for mode m

    // LAB: Compute effective attractor ξ^eff = π^T Ξ
    void compute_effective_attractor(const float* pi, float* xi_eff) const;

    const std::vector<float>& attractor_bank() const { return Xi; } // Xi = [M x Dr]

    aurelis::lens::Linear& W_pi() { return m_W_pi; }

private:
    ArcConfig cfg;
    std::vector<float> Xi; // Attractor bank: shape [num_attractors, Dr]
    std::vector<float> Lambda; // Per-attractor precision (Cholesky): [num_attractors, Dr x Dr]
    aurelis::lens::Linear m_W_pi; // Inputs [Dc + De + num_modes] -> [num_attractors]
};

} // namespace aurelis::arc
