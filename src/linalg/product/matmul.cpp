#include "matmul.hpp"
#include "kernel.hpp"
#include "../../utils/check.hpp"
#include "../../utils/broadcast.hpp"
#include "../../utils/metadata.hpp"

Tensor c_dot(const Tensor& a, const Tensor& b) {
    // (N) @ (N)
    utils::check::dotConditions(a.shape, b.shape);

    Tensor result_tensor = Tensor();

    // ----- Assigning metadata -----
    result_tensor.shape = {};
    result_tensor.ndim = 0;
    result_tensor.size = 1;
    result_tensor.strides = {1};
    result_tensor.requires_grad = a.requires_grad || b.requires_grad;
    // ------------------------------

    result_tensor._data = {0};
    for (int i = 0; i < a.shape[0]; i++) {
        result_tensor._data[0] += a._data[i] * b._data[i];
    }

    return result_tensor;
}

Tensor c_matvec(const Tensor& a, const Tensor& b) {
    // (..., M, N) @ (N)
    utils::check::matvecConditions(a.shape, b.shape);

    Tensor result_tensor = Tensor();
    utils::broadcast::BroadcastingUtilsObject BroadcastUtils(a.shape, b.shape, true); // matmul == true

    // ----- Assigning metadata -----
    result_tensor.shape = BroadcastUtils.getResultShape();
    result_tensor.ndim = utils::metadata::getNumDim(result_tensor.shape);
    result_tensor.size = utils::metadata::getSize(result_tensor.shape);
    result_tensor.strides = utils::metadata::getStrides(result_tensor.shape);
    result_tensor.requires_grad = a.requires_grad || b.requires_grad;
    // ------------------------------

    const int M = a.shape[a.ndim-2];
    const int N = b.shape[0];

    const int DATA1_ROWS = utils::factorCeilingFunc(M, BLOCK_N_ROWS);
    const int BATCH_SIZE = utils::broadcast::getBatchSize(a.shape);

    const int DATA1_MAT_SIZE = DATA1_ROWS * N;
    const int DATA2_MAT_SIZE = N * BLOCK_N_COLS;
    const int RESULT_MAT_SIZE = DATA1_MAT_SIZE * BLOCK_N_COLS;

    float* result_matrix = (float*)std::calloc(RESULT_MAT_SIZE, sizeof(float));
    float* data1_matrix = (float*)std::calloc(DATA1_MAT_SIZE, sizeof(float));
    float* data2_matrix = (float*)std::calloc(DATA2_MAT_SIZE, sizeof(float));
    if (!result_matrix || !data1_matrix || !data2_matrix) {
        throw std::runtime_error("Memory allocation failed.\n");
    }

    result_tensor._data = std::vector<float>(result_tensor.size, 0.0f);

    for (int i = 0; i < N; i++) {
        data2_matrix[i * BLOCK_N_COLS] = b._data[i];
    }
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        std::memcpy(&data1_matrix[0], &a._data[batch * a.strides[a.ndim-3]], M * N * sizeof(float));
        for (int x = 0; x < DATA1_ROWS; x += BLOCK_N_ROWS) {
            c_matmul_cpu(data1_matrix, data2_matrix, result_matrix, x, 0, 0, N, BLOCK_N_COLS, N);
        }
        for (int i = 0; i < M; i++) {
            const int IDX = batch * result_tensor.strides[result_tensor.ndim-2] + i * result_tensor.strides[result_tensor.ndim-1];
            result_tensor._data[IDX] = result_matrix[i * BLOCK_N_COLS];
        }
    }
    std::free(result_matrix);
    std::free(data1_matrix);
    std::free(data2_matrix);

    return result_tensor;
}

Tensor c_vecmat(const Tensor& a, const Tensor& b) {
    // (N) @ (..., N, P)
    utils::check::vecmatConditions(a.shape, b.shape);

    Tensor result_tensor = Tensor();
    utils::broadcast::BroadcastingUtilsObject BroadcastUtils(a.shape, b.shape, true); // matmul == true

    // ----- Assigning metadata -----
    result_tensor.shape = BroadcastUtils.getResultShape();
    result_tensor.ndim = utils::metadata::getNumDim(result_tensor.shape);
    result_tensor.size = utils::metadata::getSize(result_tensor.shape);
    result_tensor.strides = utils::metadata::getStrides(result_tensor.shape);
    result_tensor.requires_grad = a.requires_grad || b.requires_grad;
    // ------------------------------

    const int N = a.shape[0];
    const int P = b.shape[b.ndim-1];

    const int DATA2_COLS = utils::factorCeilingFunc(P, BLOCK_N_COLS);
    const int BATCH_SIZE = utils::broadcast::getBatchSize(b.shape);

    const int DATA1_MAT_SIZE = BLOCK_N_ROWS * N;
    const int DATA2_MAT_SIZE = BATCH_SIZE * N * DATA2_COLS;
    const int RESULT_MAT_SIZE = BLOCK_N_ROWS * DATA2_COLS;
    
    float* result_matrix = (float*)std::calloc(RESULT_MAT_SIZE, sizeof(float));
    float* data1_matrix = (float*)std::calloc(DATA1_MAT_SIZE, sizeof(float));
    float* data2_matrix = (float*)std::calloc(DATA2_MAT_SIZE, sizeof(float));
    if (!result_matrix || !data1_matrix || !data2_matrix) {
        throw std::runtime_error("Memory allocation failed.\n");
    }

    result_tensor._data = std::vector<float>(result_tensor.size, 0.0f);

    std::memcpy(&data1_matrix[0], &a._data[0], N * sizeof(float));
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int i = 0; i < N; i++) {
            std::memcpy(&data2_matrix[i * DATA2_COLS], &b._data[batch * b.strides[b.ndim-3] + i * b.strides[b.ndim-2]], P * sizeof(float));
        }
        for (int y = 0; y < DATA2_COLS; y += BLOCK_N_COLS) {
            c_matmul_cpu(data1_matrix, data2_matrix, result_matrix, 0, y, 0, N, DATA2_COLS, N);
        }
        std::memcpy(&result_tensor._data[batch * result_tensor.strides[result_tensor.ndim-2]], &result_matrix[0], P * sizeof(float));
    }
    std::free(result_matrix);
    std::free(data1_matrix);
    std::free(data2_matrix);

    return result_tensor;
}

Tensor c_matmul(const Tensor& a, const Tensor& b) {
    // a.shape = (..., M, N), b.shape = (..., N, P)
    utils::check::matmulConditions(a.shape, b.shape);

    Tensor result_tensor = Tensor();
    utils::broadcast::BroadcastingUtilsObject BroadcastUtils(a.shape, b.shape, true); // matmul == true
    
    // ----- Assigning metadata -----
    result_tensor.shape = BroadcastUtils.getResultShape();
    result_tensor.ndim = utils::metadata::getNumDim(result_tensor.shape);
    result_tensor.size = utils::metadata::getSize(result_tensor.shape);
    result_tensor.strides = utils::metadata::getStrides(result_tensor.shape);
    result_tensor.requires_grad = a.requires_grad || b.requires_grad;
    // ------------------------------
    
    const int M = a.shape[a.ndim - 2];
    const int N = a.shape[a.ndim - 1];
    const int P = b.shape[b.ndim - 1];
    
    const int max_ndim = result_tensor.ndim;
    
    utils::IntPtrs stridesPtrs = BroadcastUtils.getBroadcastStrides(); 
    int* stridesA = stridesPtrs.ptr1;
    int* stridesB = stridesPtrs.ptr2;
    
    const int BATCH_SIZE = utils::broadcast::getBatchSize(result_tensor.shape);
    
    const int DATA1_ROWS = utils::factorCeilingFunc(M, BLOCK_N_ROWS);
    const int DATA2_COLS = utils::factorCeilingFunc(P, BLOCK_N_COLS);
    
    const int DATA1_MAT_SIZE = DATA1_ROWS * N;
    const int DATA2_MAT_SIZE = N * DATA2_COLS;
    const int RESULT_MAT_SIZE = DATA1_ROWS * DATA2_COLS;
    
    const int NDIM_BATCH = (max_ndim > 2) ? max_ndim - 2 : 1;
    
    int* idxs1 = utils::populateLinearIdxs(result_tensor.shape, stridesA, 2);
    std::free(stridesA);
    int* idxs2 = utils::populateLinearIdxs(result_tensor.shape, stridesB, 2);
    std::free(stridesB);

    // ------------------------------------------
    float* result_matrix = (float*)std::calloc(RESULT_MAT_SIZE, sizeof(float));
    float* data1_matrix = (float*)std::calloc(DATA1_MAT_SIZE, sizeof(float));
    float* data2_matrix = (float*)std::calloc(DATA2_MAT_SIZE, sizeof(float));
    if (!result_matrix || !data1_matrix || !data2_matrix) {
        throw std::runtime_error("Memory allocation failed.\n");
    }
    
    /*
    Pads the rows of the first matrix with 0s until multiple of BLOCK_N_ROWS and
    pads the columns of the second matrix with 0s until multiple of BLOCK_N_COLS, then
    performs the matmul on the matrix using the kernel,
    before placing it in an intermediate result memory block.
    i.e. if 
    -> data1[0:6] = [1,2,3,4,5,6] and is 3x2
    -> BLOCK_N_ROWS = 6 & BLOCK_N_COLS then:
    >>  data1_matrix = [1,2,3,4,5,6,0,0,0,0,0,0]
    If data2[0:6] = [1,2,3,4,5,6] and is 2x3 then:
    >> data2_matrix = [1,2,3,0,0,0,0,0,4,5,6,0,0,0,0,0]
    then the kernel is applied to data1_matrix and data2_matrix to result in a 6x8 matrix, which is a
    block matrix of the intermediate result padded with 0s.
    */

    result_tensor._data = std::vector<float>(result_tensor.size, 0.0f);
    
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        std::memcpy(&data1_matrix[0], &a._data[idxs1[batch]], M * N * sizeof(float));
        for (int i = 0; i < N; i++) {
            std::memcpy(&data2_matrix[i * DATA2_COLS], &b._data[idxs2[batch] + i * b.strides[b.ndim-2]], P * sizeof(float));
        } 
        for (int x = 0; x < DATA1_ROWS; x += BLOCK_N_ROWS) {
            for (int y = 0; y < DATA2_COLS; y += BLOCK_N_COLS) {
                c_matmul_cpu(data1_matrix, data2_matrix, result_matrix, x, y, 0, N, DATA2_COLS, N);
            }
        }
        for (int i = 0; i < M; i++) {
            std::memcpy(&result_tensor._data[batch * M * P + i * P], &result_matrix[i * DATA2_COLS], P * sizeof(float));
        }
    }
    std::free(idxs1);
    std::free(idxs2);
    std::free(result_matrix);
    std::free(data1_matrix);
    std::free(data2_matrix);

    return result_tensor;
}