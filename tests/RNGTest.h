#ifdef _MSC_VER
#pragma once
#endif

#ifndef RNGTEST_H_46DF27B8_7FC6_4D91_A50A_C5E37E779E67
#define RNGTEST_H_46DF27B8_7FC6_4D91_A50A_C5E37E779E67

#include <cuda_runtime_api.h>
#include <iostream>
#include "WFObjectToString.h"

class RNGTest
{

public:

	__host__ int testAll()
	{
		std::cerr << "---------------------------------------------------------------------\n";
		std::cerr << "Testing random number generator...\n";

		return testRandomNumberGenerator();
	}
};

#endif //RNGTEST_H_46DF27B8_7FC6_4D91_A50A_C5E37E779E67