#pragma once

#include <vector>

namespace aurelis::lens {

struct MsspLayout {
    int D = 0;
    int num_scales = 4;
    std::vector<int> scale_index;

    static MsspLayout build(int D, int num_scales = 4);
};

}  // namespace aurelis::lens
