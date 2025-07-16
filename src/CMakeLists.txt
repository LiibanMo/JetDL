cmake_minimum_required(VERSION 3.16)
project(JetDL)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python 3.11 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

file (GLOB TENSOR_SOURCES
    "tensor/*.cpp"
)

file(GLOB LINALG_SOURCES 
    "linalg/*.cpp"
    "linalg/product/*.cpp"
)

file(GLOB UTILS_SOURCES
    "utils/*.cpp"
)

pybind11_add_module(_Cpp 
    bindings.cpp
    ${TENSOR_SOURCES}
    ${LINALG_SOURCES}
    ${UTILS_SOURCES}
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
)

target_include_directories(_Cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} 
)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/opt/homebrew/opt/openblas")
message(STATUS "CMAKE_PREFIX_PATH set to: ${CMAKE_PREFIX_PATH}")

set(BLA_VENDOR OpenBLAS)

find_package(BLAS REQUIRED)

if(BLAS_FOUND)
    message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
    message(STATUS "BLAS include directories: ${BLAS_INCLUDE_DIRS}")

    target_link_libraries(_Cpp PRIVATE 
        ${BLAS_LIBRARIES}
    )

    if(BLAS_INCLUDE_DIRS)
        target_include_directories(_Cpp PRIVATE ${BLAS_INCLUDE_DIRS})
    else()
        # Fallback if BLAS_INCLUDE_DIRS
        message(WARNING "BLAS_INCLUDE_DIRS was not set by FindBLAS. Manually adding common OpenBLAS include path.")
        target_include_directories(_Cpp PRIVATE "${CMAKE_PREFIX_PATH}/include")
    endif()
else()
    message(FATAL_ERROR "OpenBLAS (or a compatible BLAS) not found. Please install OpenBLAS and ensure it's discoverable by CMake.")
endif()