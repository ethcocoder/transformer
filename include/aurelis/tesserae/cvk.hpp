#pragma once

#include <aurelis/tensor.hpp>
#include <vector>
#include <string>

namespace aurelis::tesserae {

struct CVKConfig {
    float max_ppl_delta;     // Maximum allowed PPL delta vs teacher
    float max_ece_delta;     // Maximum allowed ECE delta vs teacher
    float max_size_mb;       // Maximum allowed model size in MB
    float max_ram_mb;        // Maximum allowed peak RAM in MB
};

class CVK {
public:
    explicit CVK(const CVKConfig& cfg);

    // FBC: Frame-based Calibration check
    bool check_calibration(const std::vector<float>& teacher_ppl,
                           const std::vector<float>& student_ppl,
                           const std::vector<float>& teacher_ece,
                           const std::vector<float>& student_ece) const;

    // SBA: Size Bound Assurance
    bool check_size(float model_size_mb) const;

    // ICB: Information Content Bound (variational MI lower bound on z)
    bool check_information_bound(const std::vector<Tensor>& layer_z, float mi_lower_bound) const;

    // Run all verification checks
    struct VerificationResult {
        bool passed;
        std::vector<std::string> errors;
    };
    VerificationResult verify(const std::vector<float>& teacher_ppl,
                              const std::vector<float>& student_ppl,
                              const std::vector<float>& teacher_ece,
                              const std::vector<float>& student_ece,
                              float model_size_mb,
                              float peak_ram_mb,
                              const std::vector<Tensor>& layer_z) const;

private:
    CVKConfig cfg_;
};

} // namespace aurelis::tesserae
