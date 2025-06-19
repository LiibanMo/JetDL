#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    float* data;
    int* shape;
    int* strides;
    int ndim;
    int size;
} Tensor;

extern "C" {
    Tensor* create_tensor(float* data, int* shape, int ndim);
    float get_item(Tensor* tensor, int* indices);
    Tensor* make_contiguous(Tensor* tensor);

    Tensor* add_tensor(Tensor* tensor, Tensor* tensorB);
    Tensor* scalar_add_tensor(Tensor* tensorA, float operand);
    Tensor* add_broadcasted(Tensor* tensorA, Tensor* tensorB);

    Tensor* sub_tensor(Tensor* tensorA, Tensor* tensorB);
    Tensor* scalar_sub_tensor(Tensor* tensorA, float operand); 
    Tensor* sub_broadcasted(Tensor* tensorA, Tensor* tensorB);

    Tensor* hadamard_mul_tensor(Tensor* tensorA, Tensor* tensorB);
    Tensor* scalar_mul_tensor(Tensor* tensorA, float operand);
    Tensor* mul_broadcasted(Tensor* tensorA, Tensor* tensorB);

    Tensor* vector_dot_product(Tensor* tensorA, Tensor* tensorB);
    Tensor* matmul_2d_2d(Tensor* tensorA, Tensor* tensorB);
    Tensor* matmul_broadcasted(Tensor* tensorA, Tensor* tensorB);

    Tensor* div_tensor(Tensor* tensorA, Tensor* tensorB);
    Tensor* scalar_div_tensor(Tensor* tensor, float divisor);
    Tensor* div_broadcasted(Tensor* tensorA, Tensor* tensorB);

    Tensor* reshape_tensor(Tensor* tensor, int* new_shape, int new_ndim);
    Tensor* flatten_tensor(Tensor* tensor);

    Tensor* transpose_tensor(Tensor* tensor);
    Tensor* matrix_transpose_tensor(Tensor* tensor);

    Tensor* sum_tensor(Tensor* tensor);
    Tensor* sum_axis_tensor(Tensor* tensor, const int axis);

    Tensor* mean_tensor(Tensor* tensor);
    Tensor* mean_axis_tensor(Tensor* tensor, const int axis);

    Tensor* pow_tensor(Tensor* tensor, float exponent);

    Tensor* ones(int* shape, const int ndim);
    Tensor* outer(Tensor* tensorA, Tensor* tensorB);
    Tensor* c_exp(Tensor* tensorA);
    Tensor* c_log(Tensor* tensor);

    void free_tensor(Tensor* tensor_ptr);
}

#endif