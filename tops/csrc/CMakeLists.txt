cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckLanguage)
include(cmake/modules/set_ifndef.cmake)
include(cmake/modules/find_library_create_target.cmake)
include(cmake/modules/resolve_dirs.cmake)
include(cmake/modules/parse_make_options.cmake)

project(tops LANGUAGES CXX)

option(FAST_BUILD "Skip compiling some kernels to accelerate compiling" OFF)

if(FAST_BUILD)
    add_compile_definitions("FAST_BUILD")
    message(WARNING "Skip some kernels to accelerate compilation")
endif()

# Determine CUDA version before enabling the language extension
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")

    if(NOT WIN32) # Linux
        execute_process(
            COMMAND
            "bash" "-c"
            "${CMAKE_CUDA_COMPILER} --version | egrep -o 'V[0-9]+.[0-9]+.[0-9]+' | cut -c2-"
            RESULT_VARIABLE _BASH_SUCCESS
            OUTPUT_VARIABLE CMAKE_CUDA_COMPILER_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE)

        if(NOT _BASH_SUCCESS EQUAL 0)
            message(FATAL_ERROR "Failed to determine CUDA version")
        endif()

    else() # Windows
        execute_process(
            COMMAND ${CMAKE_CUDA_COMPILER} --version
            OUTPUT_VARIABLE versionString
            RESULT_VARIABLE versionResult)

        if(versionResult EQUAL 0 AND versionString MATCHES
            "V[0-9]+\\.[0-9]+\\.[0-9]+")
            string(REGEX REPLACE "V" "" version ${CMAKE_MATCH_0})
            set(CMAKE_CUDA_COMPILER_VERSION "${version}")
        else()
            message(FATAL_ERROR "Failed to determine CUDA version")
        endif()
    endif()
else()
    message(FATAL_ERROR "No CUDA compiler found")
endif()

set(CUDA_REQUIRED_VERSION "11.2")

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS CUDA_REQUIRED_VERSION)
    message(
        FATAL_ERROR
        "CUDA version ${CMAKE_CUDA_COMPILER_VERSION} must be at least ${CUDA_REQUIRED_VERSION}"
    )
endif()

# Initialize CMAKE_CUDA_ARCHITECTURES before enabling CUDA
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.8")
        set(CMAKE_CUDA_ARCHITECTURES 70-real 80-real 86-real 89-real 90-real)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 70-real 80-real 86-real)
    endif()
endif()

message(STATUS "GPU architectures: ${CMAKE_CUDA_ARCHITECTURES}")

enable_language(C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

resolve_dirs(CUDAToolkit_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")

message(STATUS "CUDA library status:")
message(STATUS "    version: ${CUDAToolkit_VERSION}")
message(STATUS "    libraries: ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "    include path: ${CUDAToolkit_INCLUDE_DIRS}")

find_library(
    CUDNN_LIB cudnn
    HINTS ${CUDNN_ROOT_DIR} ${CUDAToolkit_LIBRARY_DIR}
    PATH_SUFFIXES lib64 lib lib/x64)
find_library(
    CUBLAS_LIB cublas
    HINTS ${CUDAToolkit_LIBRARY_DIR}
    PATH_SUFFIXES lib64 lib lib/stubs)
find_library(
    CUBLASLT_LIB cublasLt
    HINTS ${CUDAToolkit_LIBRARY_DIR}
    PATH_SUFFIXES lib64 lib lib/stubs)
find_library(
    CUDA_DRV_LIB cuda
    HINTS ${CUDAToolkit_LIBRARY_DIR}
    PATH_SUFFIXES stubs lib lib64 lib/stubs lib64/stubs)

set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

find_library(RT_LIB rt)

set(3RDPARTY_DIR ${PROJECT_SOURCE_DIR}/3rdparty)

if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11")
    add_definitions("-DENABLE_BF16")
    message(
        STATUS
        "CUDAToolkit_VERSION ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} is greater or equal than 11.0, enable -DENABLE_BF16 flag"
    )
endif()

if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11.8")
    add_definitions("-DENABLE_FP8")
    message(
        STATUS
        "CUDAToolkit_VERSION ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} is greater or equal than 11.8, enable -DENABLE_FP8 flag"
    )
endif()

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -DBUILD_SYSTEM=cmake_oss -DENABLE_MULTI_DEVICE=${ENABLE_MULTI_DEVICE}"
)

# Disable deprecated declarations warnings
if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations ${CMAKE_CXX_FLAGS}")
else()
    # /wd4996 is the Windows equivalent to turn off warnings for deprecated
    # declarations
    set(CMAKE_CXX_FLAGS "/wd4996 ${CMAKE_CXX_FLAGS}")
endif()

# A Windows header file defines max() and min() macros, which break our macro
# declarations.
if(WIN32)
    set(CMAKE_CXX_FLAGS "/DNOMINMAX ${CMAKE_CXX_FLAGS}")
endif()

if((MSVC))
    if((MSVC_VERSION GREATER_EQUAL 1914))
        # MSVC does not apply the correct __cplusplus version per the C++ standard
        # by default. This is required for compiling CUTLASS 3.0 kernels on windows
        # with C++-17 constexpr enabled. The 2017 15.7 MSVC adds /Zc:__cplusplus to
        # set __cplusplus to 201703 with std=c++17. See
        # https://learn.microsoft.com/en-us/cpp/build/reference/zc-cplusplus for
        # more info.
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler  /Zc:__cplusplus")
    else()
        message(
            FATAL_ERROR
            "Build is only supported with Visual Studio 2017 version 15.7 or higher"
        )
    endif()
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

if(FAST_MATH)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
    message("CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
endif()

set(COMMON_HEADER_DIRS ${CUDAToolkit_INCLUDE_DIR} ${CUDNN_ROOT_DIR}/include ${PROJECT_SOURCE_DIR} ${3RDPARTY_DIR}/cutlass/include)
message(STATUS "COMMON_HEADER_DIRS: ${COMMON_HEADER_DIRS}")

list(APPEND COMMON_HEADER_DIRS)
include_directories(${COMMON_HEADER_DIRS})

# add_subdirectory(tensorrt_llm)
add_subdirectory(tops)
add_subdirectory(example)
