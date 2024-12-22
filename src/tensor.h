#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    double* data;
    int* shape;
    int* strides;
    int ndim;
    int size;
} Tensor;

extern "C" {
    Tensor* create_tensor(double* data, int* shape, int ndim);
    double get_item(Tensor* tensor, int* indices);
    Tensor* add_tensor(Tensor* tensor, Tensor* tensorB);
    Tensor* scalar_add_tensor(Tensor* tensorA, double operand);
    Tensor* sub_tensor(Tensor* tensorA, Tensor* tensorB);
    Tensor* scalar_sub_tensor(Tensor* tensorA, double operand);
    Tensor* hadamard_mul_tensor(Tensor* tensorA, Tensor* tensorB);
    Tensor* inner_product_tensor(Tensor* tensorA, Tensor* tensorB);
    Tensor* matmul_tensor_vector(Tensor* tensorA, Tensor* tensorB);
    Tensor* matmul_tensor(Tensor* tensorA, Tensor* tensorB);
    Tensor* batch_matmul_tensor(Tensor* tensorA, Tensor* tensorB);
    Tensor* scalar_mul_tensor(Tensor* tensorA, double operand);
    Tensor* scalar_div_tensor(Tensor* tensor, double divisor);
    void free_tensor(Tensor* tensor);
    void free_data(Tensor* tensor);
    void free_shape(Tensor* tensor);
    void free_strides(Tensor* tensor);
}

#endif