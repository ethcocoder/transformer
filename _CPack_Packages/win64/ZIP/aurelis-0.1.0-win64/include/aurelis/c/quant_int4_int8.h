#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Quantize float array to INT8
void aurelis_quantize_int8(const float* in, int8_t* out, float* scale, float* zero_point, int n);

// Dequantize INT8 array back to float
void aurelis_dequantize_int8(const int8_t* in, float* out, float scale, float zero_point, int n);

// Quantize float array to INT4 (packed, 2 values per byte)
void aurelis_quantize_int4(const float* in, uint8_t* out, float* scale, float* zero_point, int n);

// Dequantize INT4 array back to float
void aurelis_dequantize_int4(const uint8_t* in, float* out, float scale, float zero_point, int n);

// Quantize with Straight-Through Estimator (STE) for training
void aurelis_quantize_ste(const float* in, float* out, float* scale, float* zero_point, int n, int bits);

#ifdef __cplusplus
}
#endif
