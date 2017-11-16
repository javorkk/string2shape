cmake_minimum_required(VERSION 2.8)
project(TilingGrammarAutoencoder)


find_package(CUDA REQUIRED)
find_package(PythonInterp 2.7 REQUIRED )
find_package(PythonLibs 2.7 REQUIRED )

include(FindCUDA)
include_directories(
    ${PYTHON_INCLUDE_DIRS}
    obj2string
)

set(CUDA_HOST_COMPILER "/usr/bin/g++-4.9")
set(CMAKE_C_COMPILER "/usr/bin/gcc-4.9")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-4.9")


list(APPEND CMAKE_CXX_FLAGS "-fPIC -std=c++11")

list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_50,code=\"sm_50,compute_50\"")


add_subdirectory (obj2string)
add_subdirectory (tests)


