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

if (WIN32)
  set_target_properties(_C PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                    ${CMAKE_SOURCE_DIR}/jetdl)
else()
  set_target_properties(_C PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                    ${CMAKE_SOURCE_DIR}/jetdl)
endif()

# ---------------- DEBUGGING STEP -----------------
get_target_property(OUTPUT_DIR _C RUNTIME_OUTPUT_DIRECTORY)
if (NOT OUTPUT_DIR)
    # Check the default property if RUNTIME_OUTPUT_DIRECTORY wasn't set successfully
    get_target_property(OUTPUT_DIR _C LIBRARY_OUTPUT_DIRECTORY)
endif()

message(STATUS "DEBUG: CMAKE_SOURCE_DIR is: ${CMAKE_SOURCE_DIR}")
message(STATUS "DEBUG: Target output property is set to: ${OUTPUT_DIR}")
message(STATUS "DEBUG: Expected full output path prefix: ${OUTPUT_DIR}/_C.pyd")
# -------------------------------------------------

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
    # FIX: On Windows, BLAS_INCLUDE_DIRS may be passed via CMAKE_ARGS
    # If not set, derive it from the library path (but handle debug/release split)
    # --------------------------------------------------------------------------------
    if (WIN32)
      message(STATUS "Windows BLAS Detection:")
      message(STATUS "  BLAS_LIBRARIES: ${BLAS_LIBRARIES}")
      
      # Check if BLAS_INCLUDE_DIRS was already set (from CMAKE_ARGS)
      if (BLAS_INCLUDE_DIRS)
        message(STATUS "  BLAS_INCLUDE_DIRS already set to: ${BLAS_INCLUDE_DIRS}")
      else()
        # Derive include path from library path
        # Note: vcpkg puts libs in debug/lib or lib, but headers always in include
        get_filename_component(BLAS_LIB_DIR "${BLAS_LIBRARIES}" DIRECTORY)
        get_filename_component(BLAS_INSTALL_ROOT "${BLAS_LIB_DIR}" DIRECTORY)
        
        # If we're in a debug directory, go up one more level
        if (BLAS_INSTALL_ROOT MATCHES "debug$")
          get_filename_component(BLAS_INSTALL_ROOT "${BLAS_INSTALL_ROOT}" DIRECTORY)
          message(STATUS "  Detected debug build, adjusted BLAS_INSTALL_ROOT")
        endif()
        
        set(VCPKG_INCLUDE_BASE "${BLAS_INSTALL_ROOT}/include")
        
        message(STATUS "  BLAS_LIB_DIR: ${BLAS_LIB_DIR}")
        message(STATUS "  BLAS_INSTALL_ROOT: ${BLAS_INSTALL_ROOT}")
        message(STATUS "  VCPKG_INCLUDE_BASE: ${VCPKG_INCLUDE_BASE}")
        
        # Test for the correct header location
        if (EXISTS "${VCPKG_INCLUDE_BASE}/cblas.h")
            set(BLAS_INCLUDE_DIRS "${VCPKG_INCLUDE_BASE}")
            message(STATUS "  Found cblas.h at: ${VCPKG_INCLUDE_BASE}/cblas.h")
        elseif (EXISTS "${VCPKG_INCLUDE_BASE}/openblas/cblas.h")
            set(BLAS_INCLUDE_DIRS "${VCPKG_INCLUDE_BASE}/openblas")
            message(STATUS "  Found cblas.h at: ${VCPKG_INCLUDE_BASE}/openblas/cblas.h")
        else()
            message(STATUS "  ERROR: cblas.h not found. Checking directory contents:")
            file(GLOB INCLUDE_FILES "${VCPKG_INCLUDE_BASE}/*")
            foreach(FILE ${INCLUDE_FILES})
              get_filename_component(FILENAME ${FILE} NAME)
              message(STATUS "    - ${FILENAME}")
            endforeach()
            
            message(FATAL_ERROR "BLAS header 'cblas.h' not found in: ${VCPKG_INCLUDE_BASE}\n"
                                "Checked:\n"
                                "  - ${VCPKG_INCLUDE_BASE}/cblas.h\n"
                                "  - ${VCPKG_INCLUDE_BASE}/openblas/cblas.h\n"
                                "Please verify vcpkg OpenBLAS installation is complete.")
        endif()
      endif()
      
      # Verify the final include directory exists and has cblas.h
      if (NOT EXISTS "${BLAS_INCLUDE_DIRS}/cblas.h")
        message(FATAL_ERROR "BLAS_INCLUDE_DIRS is set to '${BLAS_INCLUDE_DIRS}' but cblas.h not found there!")
      endif()
    endif()
    
    include_directories(${OpenBLAS_INCLUDE_DIR})
    target_link_libraries(_C PRIVATE ${BLAS_LIBRARIES})
    message(STATUS "Found BLAS libraries: ${BLAS_LIBRARIES}")
    target_include_directories(_C PRIVATE ${BLAS_INCLUDE_DIRS})
    message(STATUS "Found BLAS include directories: ${BLAS_INCLUDE_DIRS}")
  else()
    message(FATAL_ERROR "BLAS libraries not found.")
  endif()
endif()

# -------------------------------------------------------------
# FINAL STEP for Windows: Use the install() command to force copy the module
# -------------------------------------------------------------
if (WIN32)
install(TARGETS _C
    # Use LIBRARY DESTINATION for shared modules/libraries
    LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}/jetdl"
    # Use RUNTIME DESTINATION as a fallback, as PYD files are runtime components
    RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}/jetdl"
)
endif()