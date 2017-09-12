#pragma once


#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif

#include "targetver.h"

#ifdef _WIN32
//#   define WINDOWS_LEAN_AND_MEAN
#   ifndef NOMINMAX
#     define NOMINMAX 1
#   endif
#   include <windows.h>
#endif

#define _USE_MATH_DEFINES
#include <math.h>
#include <float.h>
#include <algorithm>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

#include <chrono>
#include <random>

#ifdef _WIN32
#   define FORCE_INLINE __forceinline__
#else
#   define FORCE_INLINE inline
#endif

//////////////////////////////////////////////////////////////////////////
//Useful typedefs
typedef unsigned char   byte;
typedef unsigned short  ushort;
typedef unsigned int    uint;
typedef unsigned long   ulong;

//////////////////////////////////////////////////////////////////////////
//CUDA specific includes
//////////////////////////////////////////////////////////////////////////

//#include <cuda_runtime_api.h>
//#include <vector_types.h>

//////////////////////////////////////////////////////////////////////////
//String utilities
//////////////////////////////////////////////////////////////////////////
std::string itoa(const int a);

std::string ftoa(const float a);

std::string cutComments(const std::string& aLine, const char* aToken);

std::string getDirName(const std::string& _name);