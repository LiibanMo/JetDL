#include "tensor.h"
#include "utils/metadata.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

Tensor* create_tensor(const float* _data, const size_t* shape, const size_t ndim) {
    Tensor* new_tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!new_tensor) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    
    utils_metadata_assign_basics(new_tensor, shape, ndim);
    
    new_tensor->_data = (float*)malloc(new_tensor->size * sizeof(float));
    if (!new_tensor->_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    memcpy(new_tensor->_data, _data, new_tensor->size * sizeof(float));

    return new_tensor;
}

Tensor* copy_tensor(const Tensor* src, Tensor* dest) {
    if (!src) {
        fprintf(stderr, "src is NULL.");
        return NULL;
    }

    if (!dest) {
        fprintf(stderr, "dest is NULL");
        return NULL;
    }

    if (src->_data) {
        if (dest->_data) {
            free(dest->_data);
        }
        dest->_data = (float*)malloc(src->size * sizeof(float));
        if (!dest->_data) {
            fprintf(stderr, "Memory allocation failed.\n");
            return NULL;
        }
        memcpy(dest->_data, src->_data, src->size * sizeof(float));
    } else {
        dest->_data = NULL;
    }

    if (src->shape) {
        if (dest->shape) {
            free(dest->shape);
        }
        dest->shape = (size_t*)malloc(src->size * sizeof(size_t));
        if (!dest->shape) {
            fprintf(stderr, "Memory allocation failed.\n");
            return NULL;
        }
        memcpy(dest->shape, src->shape, src->ndim * sizeof(size_t));
    } else {
        dest->shape = NULL;
    }

    dest->ndim = src->ndim;
    dest->size = src->size;

    if (src->strides) {
        if (dest->strides) {
            free(dest->strides);
        }
        dest->strides = (size_t*)malloc(dest->ndim * sizeof(size_t));
        if (!dest->strides) {
            fprintf(stderr, "Memory allocation failed.\n");
            exit(1);
        }
        memcpy(dest->strides, src->strides, dest->ndim * sizeof(size_t));
    } else {
        dest->strides = NULL;
    }

    dest->is_contiguous = src->is_contiguous;

    return dest;
}

void destroy_tensor(Tensor* tensor) {
    if (tensor->_data) {
        free(tensor->_data);
        tensor->_data = NULL;
    }
    
    if (tensor->shape) {
        free(tensor->shape);
        tensor->shape = NULL;
    }

    if (tensor->strides) {
        free(tensor->strides);
        tensor->strides = NULL;
    }
}