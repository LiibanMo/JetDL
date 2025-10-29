cmake_minimum_required(VERSION 3.16)
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
  # On other systems (e.g., Linux), use OpenBLAS
  find_package(OpenBLAS REQUIRED)

  if(OpenBLAS_FOUND)
    if(TARGET OpenBLAS::OpenBLAS)
      # Modern approach (Windows/vcpkg)
      message(STATUS "Found OpenBLAS target, linking against OpenBLAS::OpenBLAS.")
      target_link_libraries(_C PRIVATE OpenBLAS::OpenBLAS)
    else()
      # Legacy approach (Linux/apt)
      message(STATUS "OpenBLAS target not found. Assuming legacy install and linking against 'openblas' by name.")
      target_include_directories(_C PRIVATE ${OPENBLAS_INCLUDE_DIRS})
      target_link_libraries(_C PRIVATE openblas)
    endif()
  else()
    message(FATAL_ERROR "OpenBLAS not found.")
  endif()
endif()

set(OpenMP_ROOT_DIR "/opt/homebrew/opt/libomp")
find_package(OpenMP)
