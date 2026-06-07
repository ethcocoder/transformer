#include "aurelis/tesserae/p4il.hpp"

namespace aurelis::tesserae {

P4IL::P4IL(const P4ILConfig& cfg)
    : cfg_(cfg),
      inws_(cfg.inws_cfg),
      epqe_(cfg.epqe_cfg),
      efd_(cfg.efd_cfg),
      cvk_(cfg.cvk_cfg) {}

void P4IL::init() {
    inws_.init();
}

void P4IL::convert_checkpoint(const std::string& phase3_ckpt, const std::string& tesserae_dir) {
    (void)phase3_ckpt;
    (void)tesserae_dir;
    // TODO: Call Python CPCC converter script
}

void P4IL::train_warmup(const std::vector<Tensor>& target_weights) {
    (void)target_weights;
    // TODO: Implement warmup training
}

void P4IL::train_qat() {
    // TODO: Implement QAT
}

void P4IL::train_tr_prune() {
    // TODO: Implement TR pruning
}

void P4IL::bake_model(const std::string& output_path) {
    // TODO: Prepare inputs and call ODFFB::bake
    Tensor dummy_L_m, dummy_xi;
    std::vector<TRFC*> dummy_trfcs;
    odffb_.bake(output_path, inws_, layer_z_, dummy_trfcs, dummy_L_m, dummy_xi);
}

Tensor P4IL::infer(const Tensor& input) {
    (void)input;
    // TODO: Implement full inference pipeline using D3LLoader
    return Tensor::zeros({1});
}

} // namespace aurelis::tesserae
