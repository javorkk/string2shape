cmake_minimum_required(VERSION 3.2)
project(TilingGrammarAutoencoder)

find_package(CUDA REQUIRED)
find_package(PythonInterp 2.7 REQUIRED )
find_package(PythonLibs 2.7 REQUIRED )

include(FindCUDA)
include(FindOpenMP)

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    obj2string
)

set(CUDA_HOST_COMPILER "/usr/bin/g++-4.9")
set(CMAKE_C_COMPILER "/usr/bin/gcc-4.9")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-4.9")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++11")

if(OPENMP_FOUND)
    option(USE_OPENMP "Use OpenMP instead of CUDA as parallelization backend for thrust." OFF)
    if(USE_OPENMP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        list(APPEND CUDA_NVCC_FLAGS "-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP")
    endif(USE_OPENMP)
endif(OPENMP_FOUND)

#GeForce GTX 970
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_52,code=sm_52")
#Tesla P100
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_60,code=sm_60")
#GeForce GTX 1070                 
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_61,code=sm_61")
#GeForce GTX 760               
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_30,code=sm_30")



if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(ColourBold  "${Esc}[1m")
  set(Yellow      "${Esc}[33m")
  set(BoldYellow  "${Esc}[1;33m")
endif(NOT WIN32)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message( STATUS "${BoldYellow}Setting CMAKE_INSTALL_PREFIX to:   $ENV{HOME}/.local${ColourReset}")
    set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" CACHE PATH "install path" FORCE)
endif()

add_subdirectory (obj2string)
add_subdirectory (tests)
