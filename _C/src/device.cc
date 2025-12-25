#include "jetdl/device.h"

#include <stdexcept>

#ifdef JETDL_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace jetdl {

Device Device::parse(const std::string& device_str) {
  if (device_str == "cpu") {
    return Device::cpu();
  }

  if (device_str == "cuda") {
    return Device::cuda(0);
  }

  // Parse "cuda:N" format
  if (device_str.rfind("cuda:", 0) == 0) {
    std::string index_str = device_str.substr(5);
    try {
      int index = std::stoi(index_str);
      if (index < 0) {
        throw std::invalid_argument(
            "Invalid device string '" + device_str +
            "'. CUDA device index must be non-negative.");
      }
      return Device::cuda(index);
    } catch (const std::invalid_argument&) {
      throw std::invalid_argument(
          "Invalid device string '" + device_str +
          "'. Expected 'cpu', 'cuda', or 'cuda:N'.");
    } catch (const std::out_of_range&) {
      throw std::invalid_argument(
          "Invalid device string '" + device_str +
          "'. Device index out of range.");
    }
  }

  throw std::invalid_argument(
      "Invalid device string '" + device_str +
      "'. Expected 'cpu', 'cuda', or 'cuda:N'.");
}

std::string Device::str() const {
  if (type == DeviceType::CPU) {
    return "cpu";
  }
  return "cuda:" + std::to_string(index);
}

bool cuda_is_available() {
#ifdef JETDL_WITH_CUDA
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  if (error != cudaSuccess) {
    return false;
  }
  return device_count > 0;
#else
  return false;
#endif
}

int cuda_device_count() {
#ifdef JETDL_WITH_CUDA
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  if (error != cudaSuccess) {
    return 0;
  }
  return device_count;
#else
  return 0;
#endif
}

}  // namespace jetdl
