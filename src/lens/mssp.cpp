#include <cstddef>
#include "aurelis/lens/mssp.hpp"

namespace aurelis {
namespace lens {

MsspLayout MsspLayout::build(int D, int S) {
    MsspLayout layout;
    layout.scale_index.resize(static_cast<std::size_t>(D));

    for (int i = 0; i < D; ++i) {
        int s = (i * S) / D;
        layout.scale_index[static_cast<std::size_t>(i)] = s;
    }

    return layout;
}

} // namespace lens
} // namespace aurelis
