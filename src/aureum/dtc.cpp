#include "aurelis/aureum/dtc.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace aurelis::aureum {

DTC::DTC(const AureumConfig& cfg)
    : cfg_(cfg),
      centroids_(static_cast<size_t>(cfg.num_domains)),
      centroid_counts_(static_cast<size_t>(cfg.num_domains), 0.0f) {
    const auto part = aurelis::StatePartition::from_total_dim(cfg.cfg.D);
    for (auto& centroid : centroids_) {
        centroid.resize(static_cast<size_t>(part.Dm), 0.0f);
    }
}

void DTC::init() {
    // Initialize centroids to zero
    for (auto& centroid : centroids_) {
        std::fill(centroid.begin(), centroid.end(), 0.0f);
    }
    std::fill(centroid_counts_.begin(), centroid_counts_.end(), 0.0f);
}

void DTC::update_centroids(const float* m_batch, const int* domain_labels, int batch_size, int seq_len) {
    const auto part = aurelis::StatePartition::from_total_dim(cfg_.cfg.D);
    float alpha = 0.1f; // EMA decay

    for (int b = 0; b < batch_size; ++b) {
        int domain = domain_labels[b];
        if (domain < 0 || domain >= cfg_.num_domains) continue;

        // Average over sequence length
        std::vector<float> avg_m(static_cast<size_t>(part.Dm), 0.0f);
        for (int t = 0; t < seq_len; ++t) {
            const float* m_t = m_batch + (b*seq_len + t)*part.Dm;
            for (int i = 0; i < part.Dm; ++i) {
                avg_m[static_cast<size_t>(i)] += m_t[i];
            }
        }
        for (int i = 0; i < part.Dm; ++i) {
            avg_m[static_cast<size_t>(i)] /= static_cast<float>(seq_len);
        }

        // Update centroid
        auto& centroid = centroids_[static_cast<size_t>(domain)];
        centroid_counts_[static_cast<size_t>(domain)] += 1.0f;
        float w = (centroid_counts_[static_cast<size_t>(domain)] > 1.0f) ? alpha : 1.0f;
        for (int i = 0; i < part.Dm; ++i) {
            centroid[static_cast<size_t>(i)] = (1.0f - w)*centroid[static_cast<size_t>(i)] + w*avg_m[static_cast<size_t>(i)];
        }
    }
}

void DTC::compute_geodesic_direction(const float* /*m_src*/, const float* centroid_src,
                                     const float* centroid_tgt, float* v) const {
    const auto part = aurelis::StatePartition::from_total_dim(cfg_.cfg.D);
    const int Dm = part.Dm;

    // First, compute the difference vector
    std::vector<float> diff(static_cast<size_t>(Dm));
    for (int i = 0; i < Dm; ++i) {
        diff[static_cast<size_t>(i)] = centroid_tgt[i] - centroid_src[i];
    }

    // Now, for a manifold with metric G(m), the geodesic direction is a bit more involved!
    // For now, let's assume G is the identity (so manifold is Euclidean), but let's add comments for Riemannian case!
    // In the Riemannian case, we would parallel transport the vector, but for now, let's just use the normalized difference!
    float norm = 0.0f;
    for (int i = 0; i < Dm; ++i) {
        norm += diff[static_cast<size_t>(i)] * diff[static_cast<size_t>(i)];
    }
    norm = std::sqrt(norm + 1e-8f);
    for (int i = 0; i < Dm; ++i) {
        v[i] = diff[static_cast<size_t>(i)] / norm;
    }
}

void DTC::apply_parallel_transport(const float* m, const float* v, float* m_transported) const {
    const auto part = aurelis::StatePartition::from_total_dim(cfg_.cfg.D);
    const int Dm = part.Dm;

    // Again, for Euclidean space, parallel transport is just adding the vector!
    // For a Riemannian manifold, we would use the metric and connection, but for now, let's use the Euclidean version!
    // We'll add a small step size to move along the geodesic!
    float step_size = 0.1f;
    for (int i = 0; i < Dm; ++i) {
        m_transported[i] = m[i] + step_size * v[i];
    }
}

} // namespace aurelis::aureum
