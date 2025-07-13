#pragma once

#ifndef MATMUL_HPP
#define MATMUL_HPP

#include "../../tensor.hpp"

Tensor c_matmul_batched(const Tensor& a, const Tensor& b);

#endif