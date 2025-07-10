#include "product.hpp"
#include "product-cpu.hpp"

#include <pybind11/pybind11.h>
#include <vector>
#include <cstdio>
#include <algorithm>

namespace py = pybind11;

Tensor c_matmul(const Tensor& a, const Tensor& b) {
    Tensor result_tensor = Tensor();

    const int m = a.shape[a.ndim - 2];
    const int n = a.shape[a.ndim - 1];
    const int p = b.shape[b.ndim - 1];

    const int max_ndim = std::max(a.ndim, b.ndim);

    int* stridesA = (int*)malloc(max_ndim * sizeof(int));
    int* stridesB = (int*)malloc(max_ndim * sizeof(int));
    if (!stridesA || !stridesB) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    int nbatch = 1;
    for (int i = 0; i < max_ndim-2; i++) {
        int idxA = a.ndim - max_ndim + i;
        int idxB = b.ndim - max_ndim + i;

        int dimA = (idxA < 0) ? 0 : a.shape[idxA];
        int dimB = (idxB < 0) ? 0 : b.shape[idxB];

        if ((dimA != 1 && dimA < dimB) || (dimB != 1 && dimB < dimA)) {
            std::invalid_argument()
        }
        
        stridesA[i] = (dimA < dimB) ? 0 : dimA;
        stridesB[i] = (dimB < dimA) ? 0 : dimB;

        nbatch *= (dimA < dimB) ? dimB : dimA;
    }

    result_tensor.size = m * p;
    result_tensor._data = std::vector<float>(result_tensor.size, 0.0f);
    
    const int BLOCK_N_ROWS = 6;
    const int BLOCK_N_COLS = 8;

    const int DATA1_ROWS = ((m + BLOCK_N_ROWS - 1) / BLOCK_N_ROWS) * BLOCK_N_ROWS;
    const int DATA2_COLS = ((p + BLOCK_N_COLS - 1) / BLOCK_N_COLS) * BLOCK_N_COLS; 
    
    const int DATA1_SIZE = DATA1_ROWS * n;
    const int DATA2_SIZE = n * DATA2_COLS;

    float* data1 = (float*)malloc(DATA1_SIZE * sizeof(float));
    float* data2 = (float*)malloc(DATA2_SIZE * sizeof(float));
    if (!data1 || !data2) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }
    
    for (int i = 0; i < DATA1_SIZE; i++) {
        data1[i] = (i < a.size) ? a._data[i] : 0.0f;
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < DATA2_COLS; j++) {
            data2[i * DATA2_COLS + j] = (j < p) ? b._data[i * p + j] : 0.0f;
        }
    }
    
    const int RESULT_DATA_SIZE = DATA1_ROWS * DATA2_COLS;
    float* result_data = (float*)malloc(RESULT_DATA_SIZE * sizeof(float));
    if (!result_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }
    
    
    for (int x = 0; x < DATA1_ROWS; x += BLOCK_N_ROWS) {
        for (int y = 0; y < DATA2_COLS; y += BLOCK_N_COLS) {
            c_matmul_cpu(data1, data2, result_data, x, y, 0, n, DATA2_COLS, n);
        }
    }
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < DATA2_COLS; j++) {
            if (j < p) {
                result_tensor._data[i * p + j] = result_data[i * DATA2_COLS + j];
            }
        }
    }

    result_tensor.shape = {a.shape[a.ndim-2], b.shape[b.ndim-1]};
    result_tensor.ndim = 2;
    result_tensor.strides = {b.shape[b.ndim - 1], 1};
    result_tensor.requires_grad = a.requires_grad || b.requires_grad;

    free(data1);
    free(data2);
    free(result_data);

    return result_tensor;
}