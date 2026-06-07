#include "aurelis/tesserae/efd.hpp"
#include <cmath>

namespace aurelis::tesserae {

EFD::EFD(const EFDConfig& cfg) : cfg_(cfg) {}

float EFD::compute_loss(const EpistemicFrame& teacher, const EpistemicFrame& student) const {
    float loss = 0.0f;

    // FAKT loss: MSE on c, e, r, m with metric weighting
    auto compute_mse = [](const Tensor& t, const Tensor& s) {
        float mse = 0.0f;
        int n = t.numel();
        for (int i = 0; i < n; ++i) {
            float diff = t.data()[i] - s.data()[i];
            mse += diff * diff;
        }
        return mse / static_cast<float>(n);
    };

    loss += cfg_.lambda_fakt * compute_mse(teacher.c, student.c);
    loss += cfg_.lambda_fakt * compute_mse(teacher.e, student.e);
    loss += cfg_.lambda_fakt * compute_mse(teacher.r, student.r);
    loss += cfg_.lambda_fakt * compute_mse(teacher.m, student.m);
    loss += cfg_.lambda_fakt * compute_mse(teacher.kappa, student.kappa);

    if (teacher.pi.numel() > 0 && student.pi.numel() > 0) {
        loss += cfg_.lambda_abdp * compute_mse(teacher.pi, student.pi);
    }

    // TODO: Implement MMPL (Manifold Metric Preservation Loss)
    return loss;
}

} // namespace aurelis::tesserae
