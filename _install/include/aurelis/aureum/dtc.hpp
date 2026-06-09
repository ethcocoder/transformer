#pragma once

#include <aurelis/aureum/config.hpp>
#include <aurelis/tensor.hpp>
#include <aurelis/state_partition.hpp>
#include <vector>

namespace aurelis::aureum {

// Domain Transfer via Connection (DTC)
class DTC {
public:
    explicit DTC(const AureumConfig& cfg);

    void init();

    // DAC: Domain Aggregated Centroids
    void update_centroids(const float* m_batch, const int* domain_labels, int batch_size, int seq_len);

    // GFA: Geodesic Flow Alignment
    void compute_geodesic_direction(const float* m_src, const float* centroid_src,
                                     const float* centroid_tgt, float* v) const;

    // PTG: Parallel Transport Gating
    void apply_parallel_transport(const float* m, const float* v, float* m_transported) const;

    const std::vector<std::vector<float>>& centroids() const { return centroids_; }

private:
    AureumConfig cfg_;
    std::vector<std::vector<float>> centroids_;
    std::vector<float> centroid_counts_;
};

} // namespace aurelis::aureum
