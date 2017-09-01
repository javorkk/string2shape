#ifdef _MSC_VER
#pragma once
#endif

#ifndef DEBUGUTILS_H_BC4D6E0E_C335_49A2_82EA_8EA9E327A998
#define DEBUGUTILS_H_BC4D6E0E_C335_49A2_82EA_8EA9E327A998

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <iostream>
#include <iomanip>

template <typename T>
void outputDeviceVector(
	const char* aPrefix1,
	const thrust::device_vector<T>& aVec1,
	const size_t aNUM_ITEMS_TO_DISPLAY = 32)
{
	std::cerr << aPrefix1;
	if (aNUM_ITEMS_TO_DISPLAY >= (size_t)(aVec1.end() - aVec1.begin()))
	{
		thrust::copy(aVec1.begin(), aVec1.end(), std::ostream_iterator<T>(std::cerr, " "));
	}
	else
	{
		thrust::copy(aVec1.begin(), aVec1.begin() + aNUM_ITEMS_TO_DISPLAY / 2 + 1, std::ostream_iterator<T>(std::cerr, " "));
		std::cerr << " ... ";
		thrust::copy(aVec1.end() - 1 - aNUM_ITEMS_TO_DISPLAY / 2, aVec1.end(), std::ostream_iterator<T>(std::cerr, " "));
	}
	std::cerr << "\n";
}

template <typename T>
void outputHostVector(
	const char* aPrefix1,
	const thrust::host_vector<T>& aVec1,
	const size_t aNUM_ITEMS_TO_DISPLAY = 32)
{
	std::cerr << aPrefix1;
	if (aNUM_ITEMS_TO_DISPLAY >= (size_t)(aVec1.end() - aVec1.begin()))
	{
		thrust::copy(aVec1.begin(), aVec1.end(), std::ostream_iterator<T>(std::cerr, " "));
	}
	else
	{
		thrust::copy(aVec1.begin(), aVec1.begin() + aNUM_ITEMS_TO_DISPLAY / 2 + 1, std::ostream_iterator<T>(std::cerr, " "));
		std::cerr << " ... ";
		thrust::copy(aVec1.end() - 1 - aNUM_ITEMS_TO_DISPLAY / 2, aVec1.end(), std::ostream_iterator<T>(std::cerr, " "));
	}
	std::cerr << "\n";
}

class uint2_get_x
{
public:
	__host__ __device__ unsigned int operator()(const uint2 a)
	{
		return a.x;
	}
};

class uint2_get_y
{
public:
	__host__ __device__ unsigned int operator()(const uint2 a)
	{
		return a.y;
	}
};

__host__ inline std::ostream& operator<<(std::ostream& os, const uint2& aVec)
{
	os << "(" << aVec.x << ", " << aVec.y << ")";
	return os;
}

__host__ inline std::ostream& operator<<(std::ostream& os, const float3& aVec)
{
	os << std::fixed << std::setprecision(4) << "(" << aVec.x << ", " << aVec.y << ", "  << aVec.z <<")";
	return os;
}

__host__ inline std::ostream& operator<<(std::ostream& os, const quaternion4f& aVec)
{
	os << std::fixed << std::setprecision(2) << "(" << aVec.x << ", " << aVec.y << ", " << aVec.z << ", " << aVec.w << ")";
	return os;
}




#endif // DEBUGUTILS_H_BC4D6E0E_C335_49A2_82EA_8EA9E327A998
