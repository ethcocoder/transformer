#include "aurelis/c/quant_int4_int8.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

// Quantize float array to INT8
void aurelis_quantize_int8(const float* in, int8_t* out, float* scale, float* zero_point, int n) {
    float min_val = in[0];
    float max_val = in[0];
    for (int i = 1; i < n; ++i) {
        if (in[i] < min_val) min_val = in[i];
        if (in[i] > max_val) max_val = in[i];
    }

    // Symmetric quantization for simplicity
    float max_abs = fmaxf(fabsf(min_val), fabsf(max_val));
    *scale = max_abs / 127.0f;
    *zero_point = 0.0f;

    for (int i = 0; i < n; ++i) {
        float val = roundf(in[i] / *scale);
        if (val > 127.0f) val = 127.0f;
        if (val < -128.0f) val = -128.0f;
        out[i] = (int8_t)val;
    }
}

// Dequantize INT8 array back to float
void aurelis_dequantize_int8(const int8_t* in, float* out, float scale, float zero_point, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = (float)in[i] * scale + zero_point;
    }
}

// Quantize float array to INT4 (packed, 2 values per byte)
void aurelis_quantize_int4(const float* in, uint8_t* out, float* scale, float* zero_point, int n) {
    float min_val = in[0];
    float max_val = in[0];
    for (int i = 1; i < n; ++i) {
        if (in[i] < min_val) min_val = in[i];
        if (in[i] > max_val) max_val = in[i];
    }

    float max_abs = fmaxf(fabsf(min_val), fabsf(max_val));
    *scale = max_abs / 7.0f;
    *zero_point = 0.0f;

    int num_bytes = (n + 1) / 2;
    for (int i = 0; i < num_bytes; ++i) {
        int idx0 = 2 * i;
        int idx1 = 2 * i + 1;

        float val0 = roundf(in[idx0] / *scale);
        if (val0 > 7.0f) val0 = 7.0f;
        if (val0 < -8.0f) val0 = -8.0f;
        int8_t q0 = (int8_t)val0;

        uint8_t packed = 0;
        if (idx1 < n) {
            float val1 = roundf(in[idx1] / *scale);
            if (val1 > 7.0f) val1 = 7.0f;
            if (val1 < -8.0f) val1 = -8.0f;
            int8_t q1 = (int8_t)val1;
            packed = ((uint8_t)(q0 & 0x0F) << 4) | (uint8_t)(q1 & 0x0F);
        } else {
            packed = ((uint8_t)(q0 & 0x0F) << 4);
        }
        out[i] = packed;
    }
}

// Dequantize INT4 array back to float
void aurelis_dequantize_int4(const uint8_t* in, float* out, float scale, float zero_point, int n) {
    int num_bytes = (n + 1) / 2;
    for (int i = 0; i < num_bytes; ++i) {
        uint8_t packed = in[i];
        int idx0 = 2 * i;
        int idx1 = 2 * i + 1;

        int8_t q0 = (int8_t)((packed >> 4) & 0x0F);
        if (q0 & 0x08) q0 |= 0xF0; // Sign extend
        out[idx0] = (float)q0 * scale + zero_point;

        if (idx1 < n) {
            int8_t q1 = (int8_t)(packed & 0x0F);
            if (q1 & 0x08) q1 |= 0xF0;
            out[idx1] = (float)q1 * scale + zero_point;
        }
    }
}

// Quantize with Straight-Through Estimator (STE) for training
void aurelis_quantize_ste(const float* in, float* out, float* scale, float* zero_point, int n, int bits) {
    if (bits == 8) {
        int8_t* q = (int8_t*)malloc(n * sizeof(int8_t));
        aurelis_quantize_int8(in, q, scale, zero_point, n);
        aurelis_dequantize_int8(q, out, *scale, *zero_point, n);
        free(q);
    } else if (bits == 4) {
        int num_bytes = (n + 1) / 2;
        uint8_t* q = (uint8_t*)malloc(num_bytes * sizeof(uint8_t));
        aurelis_quantize_int4(in, q, scale, zero_point, n);
        aurelis_dequantize_int4(q, out, *scale, *zero_point, n);
        free(q);
    } else {
        // Fallback to FP32
        memcpy(out, in, n * sizeof(float));
        *scale = 1.0f;
        *zero_point = 0.0f;
    }
}
