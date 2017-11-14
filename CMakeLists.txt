cmake_minimum_required(VERSION 2.8)
project(TilingGrammarAutoencoder)

set(CUDA_HOST_COMPILER gcc-4.9)

find_package(CUDA REQUIRED)
find_package(PythonInterp 2.7 REQUIRED )
find_package(PythonLibs 2.7 REQUIRED )

include(FindCUDA)
include_directories(
    ${PYTHON_INCLUDE_DIRS}
    obj2string
)

list(APPEND CMAKE_CXX_FLAGS -fPIC)

list(APPEND CUDA_NVCC_FLAGS -std=c++11 -gencode=arch=compute_30,code=\"sm_30,compute_30\" -ccbin /usr/bin)


add_subdirectory (obj2string)
add_subdirectory (tests)
