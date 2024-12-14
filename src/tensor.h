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
    Tensor* add_tensor(Tensor* tensorA, Tensor* tensorB);
    Tensor* subtract_tensor(Tensor* tensorA, Tensor* tensorB);
}

#endif