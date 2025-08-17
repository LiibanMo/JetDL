#include "autograd/function.hpp"

#include <memory>

Function::Function(){
    this->_unique_identity_ptr = std::make_shared<char>();
}