#include "matmul.hpp"
#include "product-cpu.hpp"
#include "../../utils/check.hpp"
#include "../../utils/broadcast.hpp"
#include "../../utils/metadata.hpp"

Tensor c_matvec(const Tensor& a, const Tensor& b) {
    utils::check::matvecConditions(a.shape, b.shape);

    Tensor result_tensor = Tensor();
    utils::broadcast::BroadcastingUtilsObject BroadcastUtils(a.shape, b.shape, true); // matmul == true

    result_tensor.shape = BroadcastUtils.getResultShape();
    result_tensor.ndim = utils::metadata::getNumDim(result_tensor.shape);
    return result_tensor;
}

Tensor c_matmul_batched(const Tensor& a, const Tensor& b) {
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
    
    int NBATCH = 1;
    for (int i = max_ndim-3; i >= 0; i--) {
        NBATCH *= result_tensor.shape[i];
    }
    
    const int BLOCK_N_ROWS = 6;
    const int BLOCK_N_COLS = 8;
    
    const int DATA1_ROWS = ((M + BLOCK_N_ROWS - 1) / BLOCK_N_ROWS) * BLOCK_N_ROWS;
    const int DATA2_COLS = ((P + BLOCK_N_COLS - 1) / BLOCK_N_COLS) * BLOCK_N_COLS; 
    
    const int DATA1_MAT_SIZE = DATA1_ROWS * N;
    const int DATA2_MAT_SIZE = N * DATA2_COLS;
    const int RESULT_DATA_SIZE = DATA1_ROWS * DATA2_COLS;
    
    const int NDIM_BATCH = (max_ndim > 2) ? max_ndim - 2 : 1;
    
    int* max_dim_values = (int*)std::calloc(NDIM_BATCH, sizeof(int));
    if (!max_dim_values) {
        throw std::runtime_error("Memory allocation failed.\n");
    }
    /*
     subtracting 1 from each dimension from the batch dimensions.
     I.e. if result_tensor.shape = [1, 2, 2, 3, 4]
        >>> max_dim_values = [1, 2, 2] - 1
        >>> = [0, 1, 1]
    */
    
    std::transform(result_tensor.shape.begin(), result_tensor.shape.end() - 2, &max_dim_values[0], [](int x){return x - 1;});
    
    int* idxs1 = utils::populateLinearIdxs(max_dim_values, stridesA, NDIM_BATCH, NBATCH);
    std::free(stridesA);
    int* idxs2 = utils::populateLinearIdxs(max_dim_values, stridesB, NDIM_BATCH, NBATCH);
    std::free(stridesB);

    std::free(max_dim_values);
    // ------------------------------------------
    float* result_data = (float*)std::calloc(NBATCH * RESULT_DATA_SIZE, sizeof(float));
    float* result_matrix = (float*)std::calloc(RESULT_DATA_SIZE, sizeof(float));
    float* data1_matrix = (float*)std::calloc(DATA1_MAT_SIZE, sizeof(float));
    float* data2_matrix = (float*)std::calloc(DATA2_MAT_SIZE, sizeof(float));
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
    std::free(idxs1);
    std::free(idxs2);
    std::free(result_matrix);
    std::free(data1_matrix);
    std::free(data2_matrix);
    
    result_tensor._data = std::vector<float>(result_tensor.size, 0.0f);

    // populates the result_tensor._data and ignoring the padded 0s in result_data
    for (int batch = 0; batch < NBATCH; batch++) {
        for (int i = 0; i < DATA1_ROWS; i++) {
            for (int j = 0; j < DATA2_COLS; j++) {
                if (i < M && j < P) {result_tensor._data[batch * M * P + i * P + j] = result_data[batch * DATA1_ROWS * DATA2_COLS + i * DATA2_COLS + j];}
            }
        }
    }
    std::free(result_data);

    return result_tensor;
}