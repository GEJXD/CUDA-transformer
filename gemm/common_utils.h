#pragma once

#include <cstdlib>

inline void init_matrix(float* mat, int rows, int cols, float val) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = val;
    }
}

inline void init_matrix_random(float* mat, int rows, int cols, float scale = 1.0f) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX * scale;
    }
}
