#pragma once

#include <aurelis/tensor.hpp>
#include <aurelis/tesserae/inws.hpp>
#include <aurelis/tesserae/trfc.hpp>
#include <aurelis/tesserae/epqe.hpp>
#include <vector>
#include <string>
#include <fstream>

namespace aurelis::tesserae {

// D3L (Dynamic Layer Loading) for inference
class D3LLoader {
public:
    D3LLoader();

    // Load layer-specific latent codes and metadata
    void load_layer(int layer_idx);

    // Synthesize weights for current layer
    Tensor synthesize_weights(int layer_idx, const Tensor& t_emb);

    // Evict current layer from memory
    void evict_layer(int layer_idx);

    // Get peak memory usage (in MB)
    float peak_ram_mb() const;

    // Add a layer's latent code
    void add_layer_z(const Tensor& z);

    // Set the hypernetwork (INWS)
    void set_inws(INWS* inws) { inws_ = inws; }

private:
    INWS* inws_;
    std::vector<Tensor> layer_z_;
    int current_layer_;
    Tensor current_weights_;
    float peak_ram_;
};

// ODFFB (Optimized Deployment File Format) for storage
class ODFFB {
public:
    ODFFB();

    // Bake a TESSERAE model to a file
    void bake(const std::string& path,
              const INWS& inws,
              const std::vector<Tensor>& layer_z,
              const std::vector<TRFC*>& tr_cores,
              const Tensor& global_L_m,
              const Tensor& global_xi);

    // Load a TESSERAE model from a file
    void load(const std::string& path,
              INWS& inws,
              std::vector<Tensor>& layer_z,
              std::vector<TRFC*>& tr_cores,
              Tensor& global_L_m,
              Tensor& global_xi);

private:
    static const uint32_t MAGIC = 0x41555245; // "AURE" in hex
    static const uint32_t VERSION = 1;

    struct Header {
        uint32_t magic;
        uint32_t version;
        uint64_t inws_offset;
        uint64_t layer_z_offset;
        uint64_t tr_cores_offset;
        uint64_t globals_offset;
    };
};

} // namespace aurelis::tesserae
