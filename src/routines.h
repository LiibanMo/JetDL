#ifndef ROUTINES_H
#define ROUTINES_H

#include "tensor.h"

extern "C" {
    Tensor* ones(int* shape, const int ndim);
    Tensor* outer(Tensor* tensorA, Tensor* tensorB);
    Tensor* c_exp(Tensor* tensorA);
    Tensor* c_log(Tensor* tensor);
}

#endif