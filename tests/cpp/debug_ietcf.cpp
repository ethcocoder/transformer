
#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <cstdio>

namespace {
void make_orthogonal_cayley_debug(float* R, int d, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 0.02f);
    std::vector<float> S(static_cast<size_t>(d * d), 0.0f);
    fprintf(stderr, "DEBUG: Creating S matrix\n");
    for (int i = 0; i < d; ++i) {
        for (int j = i + 1; j < d; ++j) {
            const float v = dist(rng);
            S[static_cast<size_t>(i * d + j)] = v;
            S[static_cast<size_t>(j * d + i)] = -v;
        }
    }

    std::vector<float> M(static_cast<size_t>(d * d));
    std::vector<float> B(static_cast<size_t>(d * d));
    fprintf(stderr, "DEBUG: Creating M and B matrices\n");
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            const float s = S[static_cast<size_t>(i * d + j)];
            M[static_cast<size_t>(i * d + j)] = (i == j ? 1.0f : 0.0f) + s;
            B[static_cast<size_t>(i * d + j)] = (i == j ? 1.0f : 0.0f) - s;
        }
    }

    std::vector<float> Aaug(static_cast<size_t>(d * 2 * d));
    fprintf(stderr, "DEBUG: Creating Aaug matrix\n");
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            Aaug[static_cast<size_t>(i * 2 * d + j)] = M[static_cast<size_t>(i * d + j)];
            Aaug[static_cast<size_t>(i * 2 * d + d + j)] = B[static_cast<size_t>(i * d + j)];
        }
    }

    fprintf(stderr, "DEBUG: Starting Gauss-Jordan elimination, d=%d\n", d);
    for (int col = 0; col < d; ++col) {
        fprintf(stderr, "DEBUG: col=%d\n", col);
        int pivot = col;
        float maxv = std::fabs(Aaug[static_cast<size_t>(col * 2 * d + col)]);
        for (int r = col + 1; r < d; ++r) {
            const float v = std::fabs(Aaug[static_cast<size_t>(r * 2 * d + col)]);
            if (v > maxv) {
                maxv = v;
                pivot = r;
            }
        }
        fprintf(stderr, "DEBUG: pivot=%d, maxv=%g\n", pivot, maxv);
        if (maxv < 1e-12f) {
            fprintf(stderr, "DEBUG: maxv too small, using identity\n");
            for (int i = 0; i < d; ++i) {
                for (int j = 0; j < d; ++j) {
                    R[i * d + j] = (i == j) ? 1.0f : 0.0f;
                }
            }
            return;
        }
        if (pivot != col) {
            fprintf(stderr, "DEBUG: swapping rows %d and %d\n", col, pivot);
            for (int j = 0; j < 2 * d; ++j) {
                std::swap(Aaug[static_cast<size_t>(col * 2 * d + j)], Aaug[static_cast<size_t>(pivot * 2 * d + j)]);
            }
        }
        const float div = Aaug[static_cast<size_t>(col * 2 * d + col)];
        fprintf(stderr, "DEBUG: div=%g\n", div);
        if (!std::isfinite(div)) {
            fprintf(stderr, "DEBUG: div is not finite!\n");
        }
        for (int j = 0; j < 2 * d; ++j) {
            Aaug[static_cast<size_t>(col * 2 * d + j)] /= div;
        }
        for (int r = 0; r < d; ++r) {
            if (r == col) continue;
            const float factor = Aaug[static_cast<size_t>(r * 2 * d + col)];
            for (int j = 0; j < 2 * d; ++j) {
                Aaug[static_cast<size_t>(r * 2 * d + j)] -= factor * Aaug[static_cast<size_t>(col * 2 * d + j)];
            }
        }
    }

    fprintf(stderr, "DEBUG: Copying result to R\n");
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            float val = Aaug[static_cast<size_t>(i * 2 * d + d + j)];
            if (!std::isfinite(val)) {
                fprintf(stderr, "DEBUG: Aaug[%d][d+%d] = %g (not finite!)\n", i, j, val);
            }
            R[i * d + j] = val;
        }
    }
}
} // namespace

int main() {
    int d = 32;
    std::vector<float> R(static_cast<size_t>(d*d));
    std::mt19937 rng(7);
    make_orthogonal_cayley_debug(R.data(), d, rng);
    for (int i = 0; i < d*d; ++i) {
        if (!std::isfinite(R[i])) {
            fprintf(stderr, "DEBUG: R[%d] = %g (not finite!)\n", i, R[i]);
        }
    }
    return 0;
}
