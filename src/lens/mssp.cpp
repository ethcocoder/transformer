#include "aurelis/lens/mssp.hpp"

namespace aurelis::lens {

MsspLayout MsspLayout::build(int D, int num_scales) {
    MsspLayout layout;
    layout.D = D;
    layout.num_scales = num_scales;
    layout.scale_index.resize(static_cast<size_t>(D));
    const int per = D / num_scales;
    for (int i = 0; i < D; ++i) {
        int s = i / per;
        if (s >= num_scales) {
            s = num_scales - 1;
        }
        layout.scale_index[static_cast<size_t>(i)] = s;
    }
    return layout;
}

}  // namespace aurelis::lens
