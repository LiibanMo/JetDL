cmake_minimum_required(VERSION 3.18)
project(JetDL)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(
  Python 3.11
  COMPONENTS Interpreter Development
  REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB_RECURSE C_FILES "*.cc")

message(STATUS "C_FILES: ${C_FILES}")

pybind11_add_module(_C ${C_FILES})

set_target_properties(_C PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                    ${CMAKE_SOURCE_DIR}/jetdl)

option(ENABLE_DEBUG OFF)
if(ENABLE_DEBUG)
  target_compile_options(_C PRIVATE -O0 -g -no-omit-frame-pointer
                                    -fsanitize=address)
else()
  if(MSVC)
    target_compile_options(_C PRIVATE /O2 /DNDEBUG /fp:fast)
  else()
    target_compile_options(_C PRIVATE -O3 -DNDEBUG -march=native -ffast-math)
  endif()
endif()

target_include_directories(_C PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(_C PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../internal)

# --- BLAS Configuration ---
if(APPLE)
  # On macOS, use the native, hardware-optimized Accelerate framework
  target_compile_definitions(_C PRIVATE -DACCELERATE_NEW_LAPACK)
  find_library(ACCELERATE_FRAMEWORK Accelerate)
  if(ACCELERATE_FRAMEWORK)
    message(STATUS "Found Accelerate framework, linking against it.")
    target_link_libraries(_C PRIVATE ${ACCELERATE_FRAMEWORK})
  else()
    message(FATAL_ERROR "Accelerate framework not found on Apple system.")
  endif()
else()
  set(BLA_VENDOR OpenBLAS)
  find_package(BLAS REQUIRED)

  if (BLAS_FOUND)
      target_link_libraries(_C PRIVATE BLAS::BLAS)
      message(STATUS "Found BLAS libraries: ${BLAS_LIBRARIES}")
      if (WIN32)
          # 1. Manually set the paths to the vcpkg OpenBLAS installation
          # CMAKE_SOURCE_DIR is the root of your project where 'vcpkg' is cloned
          set(VCPKG_BASE_PATH "${CMAKE_SOURCE_DIR}/vcpkg/installed/x64-windows")

          # The library is in the 'lib' directory (OpenBLAS is generally static/imported)
          set(BLAS_LIBRARIES "${VCPKG_BASE_PATH}/lib/openblas.lib")

          # The headers are in the 'include' directory
          set(BLAS_INCLUDE_DIRS "${VCPKG_BASE_PATH}/include")
          target_include_directories(_C PRIVATE ${BLAS_INCLUDE_DIRS})
          message(STATUS "Found BLAS directories: ${BLAS_INCLUDE_DIRS}")
      endif()
  else()
      message(FATAL_ERROR "BLAS libraries not found.")
  endif()
endif()