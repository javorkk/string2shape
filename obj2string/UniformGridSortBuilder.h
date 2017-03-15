#ifdef _MSC_VER
#pragma once
#endif

#ifndef UNIFORMGRIDSORTBUILDER_H_7E4645DA_D20A_4889_BF6B_F87711D42CEC
#define UNIFORMGRIDSORTBUILDER_H_7E4645DA_D20A_4889_BF6B_F87711D42CEC

#include "UniformGrid.h"
#include "WFObject.h"

class UniformGridSortBuilder
{
	int resX;
	int resY;
	int resZ;
	float totalTime;
public:

	__host__ UniformGrid build(
		WFObject&						aGeometry,
		const int                       aResX,
		const int                       aResY,
		const int                       aResZ);

	__host__ int test(
		UniformGrid&		aGrid,
		WFObject&			aGeometry);

	__host__ void stats();

};

#endif // UNIFORMGRIDSORTBUILDER_H_7E4645DA_D20A_4889_BF6B_F87711D42CEC
