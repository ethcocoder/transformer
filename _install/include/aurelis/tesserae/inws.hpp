#pragma once

#include <aurelis/tensor.hpp>
#include <aurelis/lens/linear.hpp>
#include <vector>

namespace aurelis::tesserae {

struct INWSConfig {
    int latent_dim;         // Dimension of z^(l) (per-layer latent code)
    int pe_dim;             // Dimension of positional encoding for parameter index
    int layer_type_embed;   // Dimension of layer-type embedding
    int hidden_dim;         // Hidden dimension of hypernetwork MLP
};

class INWS {
public:
    explicit INWS(const INWSConfig& cfg);

    void init();

    // Synthesize weights for a single parameter index i
    // z: latent code for this layer
    // pe_i: positional encoding for parameter index i
    // t_emb: layer-type embedding (LENS, AUREUM, ARC)
    Tensor synthesize(const Tensor& z, const Tensor& pe_i, const Tensor& t_emb) const;

    // Get weights for a full weight tensor of shape [out_dim, in_dim]
    Tensor synthesize_full(const Tensor& z, const Tensor& t_emb, int out_dim, int in_dim) const;

    // Linear layers for the hypernetwork
    lens::Linear& fc1() { return fc1_; }
    lens::Linear& fc2() { return fc2_; }
    lens::Linear& fc3() { return fc3_; }

private:
    INWSConfig cfg_;
    lens::Linear fc1_;   // Input: [z; pe_i; t_emb] -> hidden
    lens::Linear fc2_;   // Hidden -> hidden
    lens::Linear fc3_;   // Hidden -> output weight
};

} // namespace aurelis::tesserae
