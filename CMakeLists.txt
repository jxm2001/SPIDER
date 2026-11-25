cmake_minimum_required(VERSION 3.18)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(SPTCConv LANGUAGES CXX CUDA)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
enable_language(CXX CUDA)

# Define the include DIRs
include_directories(third_party/argparse/include)

# 设置可执行文件的输出目录为 build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src/1d_half_sparse)
add_subdirectory(src/2d_half_dense)
add_subdirectory(src/2d_half_sparse)
add_subdirectory(src/2d_half_sparse_for_ablation)

include(CTest)
add_subdirectory(src/2d_tf32_sparse)
