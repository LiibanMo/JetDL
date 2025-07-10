cmake_minimum_required(VERSION 3.16)
project(JetDL)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python 3.11 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_Cpp 
    bindings.cpp
    tensor.cpp

    linalg/product.cpp
    linalg/product-cpu.cpp
)

set_target_properties(_Cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/jetdl
)

target_compile_options(_Cpp PRIVATE
    -std=c++17
    -O3
    -ffast-math
    -fPIC
    -DNDEBUG
    -march=native
    -I/opt/homebrew/opt/openblas/include
)

target_link_libraries(_Cpp PRIVATE 
    -L/opt/homebrew/opt/openblas/lib 
    -lopenblas
    -lblas
)

target_include_directories(_Cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    /opt/homebrew/opt/openblas/include
)
