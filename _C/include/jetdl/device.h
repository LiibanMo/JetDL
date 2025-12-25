#ifndef JETDL_DEVICE_H
#define JETDL_DEVICE_H

#include <string>

namespace jetdl {

enum class DeviceType { CPU, CUDA };

class Device {
 public:
  DeviceType type;
  int index;

  Device() : type(DeviceType::CPU), index(0) {}
  Device(DeviceType type, int index = 0) : type(type), index(index) {}

  static Device cpu() { return Device(DeviceType::CPU, 0); }

  static Device cuda(int index = 0) { return Device(DeviceType::CUDA, index); }

  static Device parse(const std::string& device_str);

  std::string str() const;

  bool is_cpu() const { return type == DeviceType::CPU; }
  bool is_cuda() const { return type == DeviceType::CUDA; }

  bool operator==(const Device& other) const {
    return type == other.type && index == other.index;
  }

  bool operator!=(const Device& other) const { return !(*this == other); }
};

bool cuda_is_available();
int cuda_device_count();

}  // namespace jetdl

#endif
