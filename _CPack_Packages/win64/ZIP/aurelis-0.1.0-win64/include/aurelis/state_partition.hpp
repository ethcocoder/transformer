#pragma once

#include <cstddef>

namespace aurelis {

struct StatePartition {
    int D = 0;
    int Dc = 0;
    int De = 0;
    int Dr = 0;
    int Dm = 0;

    static StatePartition from_total_dim(int D);

    void split(const float* h, float* c, float* e, float* r, float* m) const;
    void merge(float* h, const float* c, const float* e, const float* r,
               const float* m) const;
};

}  // namespace aurelis
