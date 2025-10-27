cmake_minimum_required(VERSION 3.16)
project(JetDL)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(
  Python 3.11
  COMPONENTS Interpreter Development
  REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB_RECURSE C_FILES "*.cc")

pybind11_add_module(_C ${C_FILES})

set_target_properties(_C PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                    ${CMAKE_SOURCE_DIR}/jetdl)

option(ENABLE_DEBUG OFF)
if(ENABLE_DEBUG)
  target_compile_options(_C PRIVATE -O0 -g -no-omit-frame-pointer
                                    -fsanitize=address)
else()
  target_compile_options(_C PRIVATE -O3 -DNDEBUG -march=native -ffast-math)
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
  # On other systems (e.g., Linux), use OpenBLAS
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/opt/homebrew/opt/openblas")
  message(STATUS "CMAKE_PREFIX_PATH set to: ${CMAKE_PREFIX_PATH}")

  set(BLA_VENDOR OpenBLAS)

  find_package(BLAS REQUIRED)

  if(BLAS_FOUND)
    message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
    message(STATUS "BLAS include directories: ${BLAS_INCLUDE_DIRS}")

    target_link_libraries(_C PRIVATE ${BLAS_LIBRARIES})

    if(BLAS_INCLUDE_DIRS)
      target_include_directories(_C PRIVATE ${BLAS_INCLUDE_DIRS})
    else()
      # Fallback if BLAS_INCLUDE_DIRS
      message(WARNING "BLAS_INCLUDE_DIRS was not set by FindBLAS.
        Manually adding common OpenBLAS include path.")
      target_include_directories(_C PRIVATE "${CMAKE_PREFIX_PATH}/include")
    endif()
  else()
    message(FATAL_ERROR "OpenBLAS (or a compatible BLAS) not found.
      Please install OpenBLAS and ensure it's discoverable by CMake.")
  endif()
endif()

set(OpenMP_ROOT_DIR "/opt/homebrew/opt/libomp")
find_package(OpenMP)
