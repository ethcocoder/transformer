#pragma once

#include <aurelis/tensor.hpp>
#include <vector>

namespace aurelis::tesserae {

struct EFDConfig {
    float lambda_fakt;    // Weight for FAKT (Frame-aligned Knowledge Transfer) loss
    float lambda_mmpl;    // Weight for MMPL (Manifold Metric Preservation Loss)
    float lambda_abdp;    // Weight for ABDP (Attractor Basin Distance Preservation)
};

class EFD {
public:
    explicit EFD(const EFDConfig& cfg);

    // Compute distillation loss between teacher and student epistemic frames
    struct EpistemicFrame {
        Tensor c;       // Content channel
        Tensor e;       // Epistemic channel
        Tensor r;       // Reasoning channel
        Tensor m;       // Meta channel
        Tensor kappa;   // Competence score
        Tensor pi;      // Attractor distribution (ARC only)
    };

    float compute_loss(const EpistemicFrame& teacher, const EpistemicFrame& student) const;

private:
    EFDConfig cfg_;
};

} // namespace aurelis::tesserae
