#ifdef _MSC_VER
#pragma once
#endif

#ifndef GRAPHTEST_H_9D0FA88C_4E33_4268_8AE9_9275AD9F8B4B
#define GRAPHTEST_H_9D0FA88C_4E33_4268_8AE9_9275AD9F8B4B

#include <cuda_runtime_api.h>
#include "WFObjectToString.h"

class GraphTest
{

public:

	__host__ void testAll(const int aSize)
	{
		testGraphConstruction(aSize);
	}
};

#endif //GRAPHTEST_H_9D0FA88C_4E33_4268_8AE9_9275AD9F8B4B

