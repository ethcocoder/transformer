#pragma once

#include <aurelis/aureum/config.hpp>
#include <aurelis/aureum/ese.hpp>
#include <aurelis/aureum/cq.hpp>
#include <aurelis/aureum/kmg.hpp>
#include <aurelis/aureum/ccc.hpp>
#include <aurelis/tensor.hpp>
#include <aurelis/state_partition.hpp>

namespace aurelis::aureum {

// Phase II Interlock (PII) — 3-pass scan scheduler
class AureumPassScheduler {
public:
    explicit AureumPassScheduler(const AureumConfig& cfg);

    void init();

    // Full 3-pass forward
    void forward(const float* c_lens, int n,
                float* e, float* kappa, float* m, float* c_mod);

    // Individual passes
    void pass1(const float* c, int n, float* e, const float* m); // ESE pass (uses m_prev)
    void pass2(const float* c, const float* e, int n, float* m); // KMG pass
    void pass3(const float* c, const float* e, const float* kappa, int n, float* c_mod); // CCC pass

    ESE& ese() { return ese_; }
    CQ& cq() { return cq_; }
    KMG& kmg() { return kmg_; }
    CCC& ccc() { return ccc_; }

private:
    AureumConfig cfg_;
    ESE ese_;
    CQ cq_;
    KMG kmg_;
    CCC ccc_;
};

} // namespace aurelis::aureum
