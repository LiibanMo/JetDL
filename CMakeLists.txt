cmake_minimum_required(VERSION 3.16)

project(JetDL)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE)
    # Find the latest macOS SDK
    execute_process(
        COMMAND xcrun --sdk macosx --show-sdk-path
        OUTPUT_VARIABLE MACOSX_SDK_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(EXISTS "${MACOSX_SDK_PATH}")
        message(STATUS "Found macOS SDK at: ${MACOSX_SDK_PATH}")
        # Add the SDK's include directory to the compile options
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isysroot ${MACOSX_SDK_PATH}")
    else()
        message(WARNING "macOS SDK not found. Please ensure Xcode Command Line Tools are installed.")
    endif()
endif()

add_subdirectory(src)