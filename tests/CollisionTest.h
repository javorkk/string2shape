#ifdef _MSC_VER
#pragma once
#endif

#ifndef COLLISIONTEST_H_77A0B860_C545_45D8_921A_0D2F0C9EE60A
#define COLLISIONTEST_H_77A0B860_C545_45D8_921A_0D2F0C9EE60A

#include <cuda_runtime_api.h>
#include <iostream>
#include "WFObjectToString.h"

class CollisionTest
{

public:

	__host__ int testAll(const char * aFilename)
	{
		std::cerr << "---------------------------------------------------------------------\n";
		std::cerr << "Collision detection test for " << aFilename << "\n";
		return testCollisionGraphConstruction(aFilename);
	}
};

#endif //COLLISIONTEST_H_77A0B860_C545_45D8_921A_0D2F0C9EE60A
