#include "kernel.hpp"
#include <omp.h>

void c_add_cpu(const float* a, const float* b, float* c, const int* shA, const int* shB, int* stA, int* stB, const int SIZE, const int NDIMA, const int NDIMB) {
    const int MAX_NDIM = (NDIMA > NDIMB) ? NDIMA : NDIMB;
    for (int i = 0; i < SIZE; i++) {
        int lin_idxA = i, lin_idxB = i;
        int iA = 0, iB = 0;
        for (int j = MAX_NDIM-1; j >= 0; j--) {
            int pos_idxA = lin_idxA % ((j < NDIMA) ? shA[j] : 1);
            int pos_idxB = lin_idxB % ((j < NDIMB) ? shB[j] : 1);
            lin_idxA /= (j < NDIMA) ? shA[j] : 1;
            lin_idxB /= (j < NDIMB) ? shB[j] : 1;
            iA += pos_idxA * stA[j];
            iB += pos_idxB * stB[j];
        }
        c[i] = a[iA] + b[iB];
    }
}