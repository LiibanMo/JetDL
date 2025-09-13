#ifndef JETDL_UTILS_OBJECTS_H
#define JETDL_UTILS_OBJECTS_H

#include <memory>
#include <vector>

class Storage {
 private:
  std::shared_ptr<std::vector<float>> data;

 public:
  void set(const std::shared_ptr<std::vector<float>>& data);
  void get() const;
};

#endif
