#ifdef _MSC_VER
#pragma once
#endif

#ifndef VARIATIONGENERATOR_H_A7DF9144_A921_43B8_9E9A_495433F1A256
#define VARIATIONGENERATOR_H_A7DF9144_A921_43B8_9E9A_495433F1A256

#include <cuda_runtime_api.h>
#include "WFObject.h"
#include "Graph.h"

class VariationGenerator
{
	float totalTime;
	float initTime;
	float samplingTime;
	float matchingTime;
	float compactionTime;
	float extractionTime;
	size_t numVariations;

public:
	static const unsigned int MAX_ALLOWED_VARIATIONS = 1000u;

	__host__ std::string operator()(
		const char * aFilePath1,
		const char * aFilePath2,
		WFObject& aObj1,
		WFObject& aObj2,
		Graph& aGraph1,
		Graph& aGraph2,
		float aRelativeThreshold);
	__host__ void stats();
};

#endif //VARIATIONGENERATOR_H_A7DF9144_A921_43B8_9E9A_495433F1A256
