#pragma once

#include <aurelis/tesserae/inws.hpp>
#include <aurelis/tesserae/trfc.hpp>
#include <aurelis/tesserae/epqe.hpp>
#include <aurelis/tesserae/eds.hpp>
#include <aurelis/tesserae/efd.hpp>
#include <aurelis/tesserae/cvk.hpp>
#include <vector>
#include <string>

namespace aurelis::tesserae {

struct P4ILConfig {
    INWSConfig inws_cfg;
    TRFCConfig trfc_cfg;
    EPQEConfig epqe_cfg;
    EFDConfig efd_cfg;
    CVKConfig cvk_cfg;
};

class P4IL {
public:
    explicit P4IL(const P4ILConfig& cfg);

    // Initialize all components
    void init();

    // Convert Phase 3 checkpoint to TESSERAE format (calls Python CPCC converter)
    void convert_checkpoint(const std::string& phase3_ckpt, const std::string& tesserae_dir);

    // Run training stage 1: Warmup (INWS weight reconstruction)
    void train_warmup(const std::vector<Tensor>& target_weights);

    // Run training stage 2: QAT (Quantization Aware Training)
    void train_qat();

    // Run training stage 3: TR pruning
    void train_tr_prune();

    // Bake final model to ODFFB
    void bake_model(const std::string& output_path);

    // Run inference on edge device
    Tensor infer(const Tensor& input);

    // Get components
    INWS& inws() { return inws_; }
    ODFFB& odffb() { return odffb_; }
    CVK& cvk() { return cvk_; }

private:
    P4ILConfig cfg_;
    INWS inws_;
    std::vector<TRFC> trfcs_;
    EPQE epqe_;
    ODFFB odffb_;
    EFD efd_;
    CVK cvk_;
    std::vector<Tensor> layer_z_;
};

} // namespace aurelis::tesserae
