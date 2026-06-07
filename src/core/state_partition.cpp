#include "aurelis/state_partition.hpp"

#include <cstring>

namespace aurelis {

StatePartition StatePartition::from_total_dim(int D) {
    StatePartition p;
    p.D = D;
    p.Dc = static_cast<int>(0.5f * static_cast<float>(D));
    p.De = static_cast<int>(0.2f * static_cast<float>(D));
    p.Dr = static_cast<int>(0.2f * static_cast<float>(D));
    p.Dm = D - p.Dc - p.De - p.Dr;
    return p;
}

void StatePartition::split(const float* h, float* c, float* e, float* r,
                           float* m) const {
    std::memcpy(c, h, static_cast<size_t>(Dc) * sizeof(float));
    std::memcpy(e, h + Dc, static_cast<size_t>(De) * sizeof(float));
    std::memcpy(r, h + Dc + De, static_cast<size_t>(Dr) * sizeof(float));
    std::memcpy(m, h + Dc + De + Dr, static_cast<size_t>(Dm) * sizeof(float));
}

void StatePartition::merge(float* h, const float* c, const float* e,
                           const float* r, const float* m) const {
    std::memcpy(h, c, static_cast<size_t>(Dc) * sizeof(float));
    std::memcpy(h + Dc, e, static_cast<size_t>(De) * sizeof(float));
    std::memcpy(h + Dc + De, r, static_cast<size_t>(Dr) * sizeof(float));
    std::memcpy(h + Dc + De + Dr, m, static_cast<size_t>(Dm) * sizeof(float));
}

}  // namespace aurelis
