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
      # --------------------------------------------------------------------------------
    # FIX: Manually derive and set the include directory path from the found library path
    # --------------------------------------------------------------------------------
    if (WIN32)
      # Use CMake path manipulation to derive the include path from the library path
      get_filename_component(BLAS_LIB_DIR "${BLAS_LIBRARIES}" DIRECTORY)
      get_filename_component(BLAS_INSTALL_ROOT "${BLAS_LIB_DIR}" DIRECTORY)
      get_filename_component(BLAS_TRIPLET_ROOT "${BLAS_INSTALL_ROOT}" DIRECTORY)

      # Set the correct include path explicitly
      set(BLAS_INCLUDE_DIRS "${BLAS_TRIPLET_ROOT}/include")
      
      message(STATUS "Corrected BLAS include directories: ${BLAS_INCLUDE_DIRS}")

      if (NOT EXISTS "${BLAS_INCLUDE_DIRS}/cblas.h")
          message(FATAL_ERROR "BLAS header 'cblas.h' not found in derived path: ${BLAS_INCLUDE_DIRS}")
      endif()
    endif()
    
    target_link_libraries(_C PRIVATE ${BLAS_LIBRARIES})
    message(STATUS "Found BLAS libraries: ${BLAS_LIBRARIES}")
    target_include_directories(_C PRIVATE ${BLAS_INCLUDE_DIRS})
    message(STATUS "Found BLAS include directories: ${BLAS_INCLUDE_DIRS}")
  else()
    message(FATAL_ERROR "BLAS libraries not found.")
  endif()
endif()