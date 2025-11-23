#include "jetdl/utils/device.h"

#ifdef JETDL_WITH_CUDA
#include <cuda_runtime.h>
#endif

bool is_cuda_available() {
#ifdef JETDL_WITH_CUDA
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        // This can happen if the driver is not installed or is the wrong version.
        return false;
    }
    return device_count > 0;
#else
    return false;
#endif
}
