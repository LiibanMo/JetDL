#ifndef UTILS_AUXILIARY
#define UTILS_AUXILIARY

#include <stddef.h>

#define UTILS_CHECK_ALLOC_FAILURE(ptr, err, goto_label) do {\
    if (!ptr) { \
        fprintf(err, "Memory allocation failed.\n"); \
        goto goto_label; \
    } \
} while(0)

#define UTILS_FREE(ptr) do {\
    free(ptr); \
    ptr = NULL; \
} while(0)

#define UTILS_GET_MAX(a, b) ((a > b) ? a : b)

#define UTILS_NEXT_MULTIPLE(input, factor) (((input + factor - 1) / factor) * factor)

typedef enum {
    MATMUL,
    DOT,
    ARITHMETIC,
} OpType;

#ifdef __cplusplus
extern "C" {
#endif

    size_t utils_get_count(
        const void* data, const void* input, const size_t N, const size_t type_size
    );
    size_t* utils_populate_linear_idxs(
        const size_t* shape, const size_t* strides, const size_t ndim, const OpType optype
    );
    size_t* utils_make_axes_positive(
        const int* axes, const size_t axes_ndim, const size_t ndim
    );
    void utils_erase_at_idx(
        void** input_ptr, const size_t idx, const size_t N, const size_t type_size
    );
    void utils_reverse(
        void** input_ptr, const size_t START, const size_t END, const size_t type_size
    ); 

#ifdef __cplusplus
}
#endif

#endif