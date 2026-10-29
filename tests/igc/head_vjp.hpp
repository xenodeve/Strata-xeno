#pragma once
// Offline training-seam experiment only. Returns W_dequant^T * delta for frozen
// Q8_0 head weights. Treating this as a native-head activation gradient uses an
// explicit straight-through estimator across Q8_1, not its exact derivative.
void igc_head_weight_vjp(const void* weights, const float* delta, float* result,
                         int width, int rows, void* stream);
