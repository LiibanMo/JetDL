#include "matmul.hpp"
#include "kernel.hpp"
#include "utils/check.hpp"
#include "utils/broadcast.hpp"
#include "utils/metadata.hpp"
#include <memory>

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

    result_tensor._data = std::shared_ptr<float[]>(new float[1]());
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

    std::unique_ptr<float[]> result_matrix(new float[RESULT_MAT_SIZE]());
    std::unique_ptr<float[]> data1_matrix(new float[DATA1_MAT_SIZE]());
    std::unique_ptr<float[]> data2_matrix(new float[DATA2_MAT_SIZE]());

    result_tensor._data = std::shared_ptr<float[]>(new float[result_tensor.size]());

    for (int i = 0; i < N; i++) {
        data2_matrix[i * BLOCK_N_COLS] = b._data[i];
    }
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        std::copy(a._data.get() + (batch * a.strides[a.ndim-3]), a._data.get() + (batch * a.strides[a.ndim-3]) + (M * N), data1_matrix.get());
        for (int x = 0; x < DATA1_ROWS; x += BLOCK_N_ROWS) {
            c_matmul_cpu(data1_matrix.get(), data2_matrix.get(), result_matrix.get(), x, 0, 0, N, BLOCK_N_COLS, N);
        }
        for (int i = 0; i < M; i++) {
            const int IDX = batch * result_tensor.strides[result_tensor.ndim-2] + i * result_tensor.strides[result_tensor.ndim-1];
            result_tensor._data[IDX] = result_matrix[i * BLOCK_N_COLS];
        }
    }

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
    
    std::unique_ptr<float[]> result_matrix(new float[RESULT_MAT_SIZE]());
    std::unique_ptr<float[]> data1_matrix(new float[DATA1_MAT_SIZE]());
    std::unique_ptr<float[]> data2_matrix(new float[DATA2_MAT_SIZE]());

    result_tensor._data = std::shared_ptr<float[]>(new float[result_tensor.size]());

    std::copy(a._data.get(), a._data.get() + N, data1_matrix.get());
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int i = 0; i < N; i++) {
            std::copy(b._data.get() + batch * b.strides[b.ndim-3] + i * b.strides[b.ndim-2], b._data.get() + batch * b.strides[b.ndim-3] + i * b.strides[b.ndim-2] + P, data2_matrix.get() + i * DATA2_COLS);
        }
        for (int y = 0; y < DATA2_COLS; y += BLOCK_N_COLS) {
            c_matmul_cpu(data1_matrix.get(), data2_matrix.get(), result_matrix.get(), 0, y, 0, N, DATA2_COLS, N);
        }
        std::copy(result_matrix.get(), result_matrix.get() + P, result_tensor._data.get() + batch * result_tensor.strides[result_tensor.ndim-2]);
    }

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
    
    const int BATCH_SIZE = utils::broadcast::getBatchSize(result_tensor.shape);
    
    const int DATA1_ROWS = utils::factorCeilingFunc(M, BLOCK_N_ROWS);
    const int DATA2_COLS = utils::factorCeilingFunc(P, BLOCK_N_COLS);
    
    const int DATA1_MAT_SIZE = DATA1_ROWS * N;
    const int DATA2_MAT_SIZE = N * DATA2_COLS;
    const int RESULT_MAT_SIZE = DATA1_ROWS * DATA2_COLS;
    
    const int NDIM_BATCH = (max_ndim > 2) ? max_ndim - 2 : 1;

    utils::IntPtrs stridesPtrs = BroadcastUtils.getBroadcastStrides(); 

    std::unique_ptr<int[]> stridesA = std::move(stridesPtrs.ptr1);
    std::unique_ptr<int[]> idxs1 = utils::populateLinearIdxs(result_tensor.shape, stridesA.get(), 2);
    std::unique_ptr<int[]> stridesB = std::move(stridesPtrs.ptr2);
    std::unique_ptr<int[]> idxs2 = utils::populateLinearIdxs(result_tensor.shape, stridesB.get(), 2);

    // ------------------------------------------
    std::unique_ptr<float[]> result_matrix (new float[RESULT_MAT_SIZE]());
    std::unique_ptr<float[]> data1_matrix(new float[DATA1_MAT_SIZE]());
    std::unique_ptr<float[]> data2_matrix(new float[DATA2_MAT_SIZE]());

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

    result_tensor._data = std::shared_ptr<float[]>(new float[result_tensor.size]());
    
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        std::copy(a._data.get() + idxs1[batch], a._data.get() + idxs1[batch] + M * N, data1_matrix.get());
        for (int i = 0; i < N; i++) {
            std::copy(b._data.get() + idxs2[batch] + i * b.strides[b.ndim-2], b._data.get() + idxs2[batch] + i * b.strides[b.ndim-2] + P, data2_matrix.get() + i * DATA2_COLS);
        } 
        for (int x = 0; x < DATA1_ROWS; x += BLOCK_N_ROWS) {
            for (int y = 0; y < DATA2_COLS; y += BLOCK_N_COLS) {
                c_matmul_cpu(data1_matrix.get(), data2_matrix.get(), result_matrix.get(), x, y, 0, N, DATA2_COLS, N);
            }
        }
        for (int i = 0; i < M; i++) {
            std::copy(result_matrix.get() + i * DATA2_COLS, result_matrix.get() + i * DATA2_COLS + P, result_tensor._data.get() + (batch * M * P + i * P));
        }
    }

    return result_tensor;
}