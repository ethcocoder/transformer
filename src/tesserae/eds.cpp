#include "aurelis/tesserae/eds.hpp"

#include <cstring>
#include <iostream>

namespace aurelis::tesserae {

// D3LLoader implementation
D3LLoader::D3LLoader() : inws_(nullptr), current_layer_(-1), peak_ram_(0.0f) {}

void D3LLoader::add_layer_z(const Tensor& z) {
    layer_z_.push_back(z);
}

void D3LLoader::load_layer(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= static_cast<int>(layer_z_.size())) {
        return;
    }
    current_layer_ = layer_idx;
}

Tensor D3LLoader::synthesize_weights(int layer_idx, const Tensor& t_emb) {
    if (!inws_ || layer_idx < 0 || layer_idx >= static_cast<int>(layer_z_.size())) {
        return Tensor::zeros({1});
    }

    // For simplicity, let's synthesize a [1024, 1024] weight tensor
    Tensor weights = inws_->synthesize_full(layer_z_[layer_idx], t_emb, 1024, 1024);

    // Track peak RAM
    float ram_mb = static_cast<float>(weights.numel() * sizeof(float)) / (1024.0f * 1024.0f);
    if (ram_mb > peak_ram_) {
        peak_ram_ = ram_mb;
    }

    current_weights_ = weights;
    return weights;
}

void D3LLoader::evict_layer(int layer_idx) {
    (void)layer_idx;
    current_weights_ = Tensor(); // Free memory
}

float D3LLoader::peak_ram_mb() const {
    return peak_ram_;
}

// ODFFB implementation
ODFFB::ODFFB() {}

void ODFFB::bake(const std::string& path,
                 const INWS& inws,
                 const std::vector<Tensor>& layer_z,
                 const std::vector<TRFC*>& tr_cores,
                 const Tensor& global_L_m,
                 const Tensor& global_xi) {
    (void)inws; (void)layer_z; (void)tr_cores; (void)global_L_m; (void)global_xi;

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << path << std::endl;
        return;
    }

    Header header;
    header.magic = MAGIC;
    header.version = VERSION;
    // TODO: Fill offsets correctly
    header.inws_offset = sizeof(Header);
    header.layer_z_offset = header.inws_offset;
    header.tr_cores_offset = header.layer_z_offset;
    header.globals_offset = header.tr_cores_offset;

    file.write(reinterpret_cast<const char*>(&header), sizeof(Header));

    // TODO: Write actual data
    file.close();
}

void ODFFB::load(const std::string& path,
                 INWS& inws,
                 std::vector<Tensor>& layer_z,
                 std::vector<TRFC*>& tr_cores,
                 Tensor& global_L_m,
                 Tensor& global_xi) {
    (void)inws; (void)layer_z; (void)tr_cores; (void)global_L_m; (void)global_xi;

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << path << std::endl;
        return;
    }

    Header header;
    file.read(reinterpret_cast<char*>(&header), sizeof(Header));

    if (header.magic != MAGIC) {
        std::cerr << "Invalid magic number in ODFFB file" << std::endl;
        return;
    }
    if (header.version != VERSION) {
        std::cerr << "Unsupported ODFFB version: " << header.version << std::endl;
        return;
    }

    // TODO: Read actual data
    file.close();
}

} // namespace aurelis::tesserae
