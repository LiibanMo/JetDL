#ifndef ROUTINES_H
#define ROUTINES_H

#include "tensor.h"

extern "C" {
    Tensor* ones(int* shape, const int ndim);
    Tensor* outer(Tensor* tensorA, Tensor* tensorB);
}

#endif