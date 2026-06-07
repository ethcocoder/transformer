#include "aurelis/tesserae/cvk.hpp"
#include <cmath>
#include <algorithm>

namespace aurelis::tesserae {

CVK::CVK(const CVKConfig& cfg) : cfg_(cfg) {}

bool CVK::check_calibration(const std::vector<float>& teacher_ppl,
                            const std::vector<float>& student_ppl,
                            const std::vector<float>& teacher_ece,
                            const std::vector<float>& student_ece) const {
    if (teacher_ppl.size() != student_ppl.size() || teacher_ece.size() != student_ece.size()) {
        return false;
    }

    float avg_ppl_delta = 0.0f;
    for (size_t i = 0; i < teacher_ppl.size(); ++i) {
        avg_ppl_delta += std::fabs(teacher_ppl[i] - student_ppl[i]);
    }
    avg_ppl_delta /= static_cast<float>(teacher_ppl.size());

    float avg_ece_delta = 0.0f;
    for (size_t i = 0; i < teacher_ece.size(); ++i) {
        avg_ece_delta += std::fabs(teacher_ece[i] - student_ece[i]);
    }
    avg_ece_delta /= static_cast<float>(teacher_ece.size());

    return avg_ppl_delta < cfg_.max_ppl_delta && avg_ece_delta < cfg_.max_ece_delta;
}

bool CVK::check_size(float model_size_mb) const {
    return model_size_mb <= cfg_.max_size_mb;
}

bool CVK::check_information_bound(const std::vector<Tensor>& layer_z, float mi_lower_bound) const {
    // TODO: Implement proper variational MI lower bound calculation
    (void)layer_z;
    (void)mi_lower_bound;
    return true;
}

CVK::VerificationResult CVK::verify(const std::vector<float>& teacher_ppl,
                                    const std::vector<float>& student_ppl,
                                    const std::vector<float>& teacher_ece,
                                    const std::vector<float>& student_ece,
                                    float model_size_mb,
                                    float peak_ram_mb,
                                    const std::vector<Tensor>& layer_z) const {
    VerificationResult result;
    result.passed = true;

    if (!check_calibration(teacher_ppl, student_ppl, teacher_ece, student_ece)) {
        result.passed = false;
        result.errors.push_back("Calibration check failed: PPL or ECE delta exceeds threshold");
    }

    if (!check_size(model_size_mb)) {
        result.passed = false;
        result.errors.push_back("Size check failed: model size exceeds " + std::to_string(cfg_.max_size_mb) + " MB");
    }

    if (peak_ram_mb > cfg_.max_ram_mb) {
        result.passed = false;
        result.errors.push_back("RAM check failed: peak RAM exceeds " + std::to_string(cfg_.max_ram_mb) + " MB");
    }

    if (!check_information_bound(layer_z, 0.0f)) {
        result.passed = false;
        result.errors.push_back("Information bound check failed");
    }

    return result;
}

} // namespace aurelis::tesserae
