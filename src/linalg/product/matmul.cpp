#include "matmul.hpp"
#include "product-cpu.hpp"

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

Tensor c_matmul_batched(const Tensor& a, const Tensor& b) {
    Tensor result_tensor = Tensor();

    if (a.shape[a.ndim-1] != b.shape[b.ndim-2]) {
        py::value_error(py::str("Invalid shapes for matmul: {} and {}.").format(a.shape, b.shape));
    }
    const int M = a.shape[a.ndim - 2];
    const int N = a.shape[a.ndim - 1];
    const int P = b.shape[b.ndim - 1];

    const int max_ndim = std::max(a.ndim, b.ndim);

    result_tensor.ndim = max_ndim;
    result_tensor.shape = std::vector<int>(max_ndim, 0);
    _obtain_strides(result_tensor);

    result_tensor.shape[max_ndim-1] = P;
    result_tensor.shape[max_ndim-2] = M; 

    int* stridesA = (int*)calloc(max_ndim, sizeof(int));
    int* stridesB = (int*)calloc(max_ndim, sizeof(int));
    if (!stridesA || !stridesB) {
        throw std::runtime_error("Memory allocation failed.\n");
    }

    stridesA[max_ndim-2] = a.strides[a.ndim-2];
    stridesB[max_ndim-2] = b.strides[b.ndim-2];
    
    stridesA[max_ndim-1] = a.strides[a.ndim-1];
    stridesB[max_ndim-1] = b.strides[b.ndim-1];

    int NBATCH = 1;
    int strideA = stridesA[max_ndim-2];
    int strideB = stridesB[max_ndim-2];
    for (int i = max_ndim-3; i >= 0; i--) {
        const int idxA = i - max_ndim + a.ndim;
        const int idxB = i - max_ndim + b.ndim;
        
        const int dimA = (idxA < 0) ? 0 : a.shape[idxA];
        const int dimB = (idxB < 0) ? 0 : b.shape[idxB];

        result_tensor.shape[i] = std::max(dimA, dimB);
        NBATCH *= result_tensor.shape[i];

        strideA *= a.shape[idxA+1];
        strideB *= b.shape[idxB+1];

        stridesA[i] = (dimA < dimB && dimA <= 1) ? 0 : strideA;
        stridesB[i] = (dimB < dimA && dimB <= 1) ? 0 : strideB;
    }
    
    result_tensor.size = NBATCH * M * P;
    result_tensor._data = std::vector<float>(result_tensor.size, 0.0f);
    
    const int BLOCK_N_ROWS = 6;
    const int BLOCK_N_COLS = 8;

    const int DATA1_ROWS = ((M + BLOCK_N_ROWS - 1) / BLOCK_N_ROWS) * BLOCK_N_ROWS;
    const int DATA2_COLS = ((P + BLOCK_N_COLS - 1) / BLOCK_N_COLS) * BLOCK_N_COLS; 
    
    const int DATA1_MAT_SIZE = DATA1_ROWS * N;
    const int DATA2_MAT_SIZE = N * DATA2_COLS;
    const int RESULT_DATA_SIZE = DATA1_ROWS * DATA2_COLS;

    const int NDIM_BATCH = max_ndim - 2;
    int* idxs1 = (int*)calloc(NBATCH, sizeof(int));
    int* idx1 = (int*)calloc(NDIM_BATCH, sizeof(int));
    int* idxs2 = (int*)calloc(NBATCH, sizeof(int));
    int* idx2 = (int*)calloc(NDIM_BATCH, sizeof(int));
    int* max_dim_values = (int*)calloc(NDIM_BATCH, sizeof(int));
    if (!idxs1 || !idx1 || !idxs2 || !idx2) {
        throw std::runtime_error("Memory allocation failed.\n");
    }

    /*
     subtracting 1 from each dimension from the batch dimensions.
     I.e. if result_tensor.shape = [1, 2, 2, 3, 4]
        >>> max_dim_values = [1, 2, 2] - 1
        >>> = [0, 1, 1]
    */
    std::transform(result_tensor.shape.begin(), result_tensor.shape.end() - 2, &max_dim_values[0], [](int x){return x - 1;});
    
    // NOTE: Repeated logic here -> need to clean.
    // ------------------------------------------
    for (int i = 0; i < NBATCH; i++) {
        for (int j = 0; j < NDIM_BATCH; j++) {
            idxs1[i] += stridesA[j] * idx1[j];
        }
        if (std::equal(idx1, idx1 + NDIM_BATCH, max_dim_values)) {
            break;
        }
        for (int axis = NDIM_BATCH-1; axis >= 0; axis--) {
            idx1[axis]++;
            if (idx1[axis] <= max_dim_values[axis]) {
                break;
            }
            idx1[axis] = 0;
        }
    }
    free(stridesA);
    free(idx1);

    for (int i = 0; i < NBATCH; i++) {
        for (int j = 0; j < NDIM_BATCH; j++) {
            idxs2[i] += stridesB[j] * idx2[j];
        }
        if (std::equal(idx2, idx2 + NDIM_BATCH, max_dim_values)) {
            break;
        }
        for (int axis = NDIM_BATCH-1; axis >= 0; axis--) {
            idx2[axis]++;
            if (idx2[axis] <= max_dim_values[axis]) {
                break;
            }
            idx2[axis] = 0;
        }
    }
    free(stridesB);
    free(idx2);
    free(max_dim_values);
    // ------------------------------------------

    float* result_data = (float*)calloc(NBATCH * RESULT_DATA_SIZE, sizeof(float));
    float* result_matrix = (float*)calloc(RESULT_DATA_SIZE, sizeof(float));
    float* data1_matrix = (float*)calloc(DATA1_MAT_SIZE, sizeof(float));
    float* data2_matrix = (float*)calloc(DATA2_MAT_SIZE, sizeof(float));
    if (!result_data || !result_matrix || !data1_matrix || !data2_matrix) {
        throw std::runtime_error("Memory allocation failed.\n");
    }
    
    /*
    Pads the rows of the first matrix with 0s until multiple of BLOCK_N_ROWS and
    pads the columns of the second matrix with 0s until multiple of BLOCK_N_COLS, then
    performs the matmul on the matrix using the kernel,
    before placing it in an intermediate result memory block.
    i.e. if data1[0:6] = [1,2,3,4,5,6] and is 3x2 then:
    >>  data1_matrix = [1,2,3,4,5,6,0,0,0,0,0,0]
    If data2[0:6] = [1,2,3,4,5,6] and is 2x3 then:
    >> data2_matrix = [1,2,3,0,0,0,0,0,4,5,6,0,0,0,0,0]
    then the kernel is applied to data1_matrix and data2_matrix to result in a 6x8 matrix, which is a
    block matrix of the intermediate result padded with 0s.
    */
    for (int batch = 0; batch < NBATCH; batch++) {
        std::memcpy(&data1_matrix[0], &a._data[idxs1[batch]], DATA1_MAT_SIZE * sizeof(float));
        for (int i = 0; i < N; i++) {
            std::memcpy(&data2_matrix[i * DATA2_COLS], &b._data[idxs2[batch] + i * P], P * sizeof(float));
        } 
        for (int x = 0; x < DATA1_ROWS; x += BLOCK_N_ROWS) {
            for (int y = 0; y < DATA2_COLS; y += BLOCK_N_COLS) {
                c_matmul_cpu(data1_matrix, data2_matrix, result_matrix, x, y, 0, N, DATA2_COLS, N);
            }
        }
        std::memcpy(&result_data[batch * RESULT_DATA_SIZE], result_matrix, RESULT_DATA_SIZE * sizeof(float));
    }
    free(idxs1);
    free(idxs2);
    free(result_matrix);
    free(data1_matrix);
    free(data2_matrix);
    
    for (int batch = 0; batch < NBATCH; batch++) {
        for (int i = 0; i < DATA1_ROWS; i++) {
            for (int j = 0; j < DATA2_COLS; j++) {
                if (i < M && j < P) {result_tensor._data[batch * M * P + i * P + j] = result_data[batch * DATA1_ROWS * DATA2_COLS + i * DATA2_COLS + j];}
            }
        }
    }
    free(result_data);

    result_tensor.requires_grad = a.requires_grad || b.requires_grad;

    return result_tensor;
}