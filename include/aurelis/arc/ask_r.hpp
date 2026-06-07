#pragma once

#include <aurelis/arc/config.hpp>
#include <aurelis/arc/ale.hpp>
#include <aurelis/arc/rcvg.hpp>
#include <aurelis/arc/mc_fws.hpp>
#include <aurelis/arc/rse.hpp>
#include <aurelis/arc/ccrc.hpp>
#include <aurelis/arc/roi.hpp>
#include <aurelis/aureum/pii.hpp>

namespace aurelis::arc {

// ASK-R: Four-pass scheduler (Passes 1-4)
class ArcPassScheduler {
public:
    explicit ArcPassScheduler(const ArcConfig& cfg);

    void init();

    // Full 4-pass forward
    void forward(const float* c_lens, const float* gamma_lens, const float* x_embed,
                int n, float* e, float* r, float* m, float* logits, float* aux_loss);

    ALE& ale() { return _ale; }
    RCVG& rcvg() { return _rcvg; }
    MCFWS& mc_fws() { return _mc_fws; }
    RSE& rse() { return _rse; }
    CCRC& ccrc() { return _ccrc; }
    ROI& roi() { return _roi; }
    aurelis::aureum::AureumPassScheduler& aureum() { return _aureum; }

private:
    ArcConfig cfg;
    ALE _ale;
    RCVG _rcvg;
    MCFWS _mc_fws;
    RSE _rse;
    CCRC _ccrc;
    ROI _roi;
    aurelis::aureum::AureumPassScheduler _aureum;
};

} // namespace aurelis::arc
