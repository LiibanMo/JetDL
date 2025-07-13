#pragma once

#ifndef PRODUCT_HPP
#define PRODUCT_HPP

#include "../tensor.hpp"

// Tensor c_dot(Tensor& a, Tensor& b);
Tensor c_matmul(Tensor& a, Tensor& b);

#endif