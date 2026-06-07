#pragma once

#include <aurelis/aureum/config.hpp>
#include <aurelis/lens/linear.hpp>
#include <aurelis/tensor.hpp>
#include <aurelis/state_partition.hpp>

namespace aurelis::aureum {

// Knowledge Metric Geometry (KMG)
class KMG {
public:
    explicit KMG(const AureumConfig& cfg);

    void init();

    // Forward step
    void forward_step(const float* c_t, const float* e_t, float* m_t) const;

    void forward_sequence(const float* c, const float* e, int n, float* m) const;

    // Compute metric tensor G_m from L_m
    void compute_Gm(const float* Lm, float* Gm, int Dm) const;

    // Geodesic distance between two points on manifold
    float geodesic_distance(const float* d1, const float* d2, const float* Gm, int Dm) const;

    // Cholesky decomposition for G_m
    aurelis::lens::Linear& W_Lm() { return W_Lm_; }

private:
    AureumConfig cfg_;
    aurelis::lens::Linear W_Lm_;
};

} // namespace aurelis::aureum
