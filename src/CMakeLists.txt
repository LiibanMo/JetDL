cmake_minimum_required(VERSION 3.16)
project(JetDL)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python 3.11 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

find_package(OpenBLAS CONFIG QUIET)
if(NOT OpenBLAS_FOUND)
    message(STATUS "OpenBLASConfig.cmake not found. Falling back to generic FindBLAS.")
    find_package(BLAS REQUIRED)
    set(ACTUAL_BLAS_LIBRARIES ${BLAS_LIBRARIES})
    set(ACTUAL_BLAS_INCLUDE_DIRS ${BLAS_INCLUDE_DIRS})
else()
    message(STATUS "OpenBLASConfig.cmake found. Using OpenBLAS specific variables.")
    set(ACTUAL_BLAS_LIBRARIES ${OpenBLAS_LIBRARIES})
    set(ACTUAL_BLAS_INCLUDE_DIRS ${OpenBLAS_INCLUDE_DIRS})
endif()

file(GLOB LINALG_SOURCES 
    "linalg/*.cpp"
    "linalg/product/*.cpp"
)

pybind11_add_module(_Cpp 
    bindings.cpp
    tensor.cpp
    ${LINALG_SOURCES}
)

set_target_properties(_Cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/jetdl
)

target_compile_options(_Cpp PRIVATE
    -std=c++17
    # -O3
    -ffast-math
    -fPIC
    # -DNDEBUG
    -g
    -march=native
)

target_link_libraries(_Cpp PRIVATE 
    ${ACTUAL_BLAS_LIBRARIES}
)

target_include_directories(_Cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ACTUAL_BLAS_INCLUDE_DIRS} 
)
