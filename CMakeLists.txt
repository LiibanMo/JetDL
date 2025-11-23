cmake_minimum_required(VERSION 3.18)

find_package(CUDAToolkit QUIET)

set(_USE_CUDA_BUILD FALSE)
set(PROJECT_LANGUAGES CXX)

if(CUDAToolkit_FOUND AND CUDAToolkit_NVCC_EXECUTABLE)
    message(STATUS "CUDA Toolkit found, enabling CUDA language for project.")
    list(APPEND PROJECT_LANGUAGES CUDA)
    set(_USE_CUDA_BUILD TRUE)
else()
    message(STATUS "CUDA Toolkit not found. Project will be CPU only.")
endif()

project(JetDL LANGUAGES ${PROJECT_LANGUAGES})

add_subdirectory(_C)
