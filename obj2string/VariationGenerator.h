#ifdef _MSC_VER
#pragma once
#endif

#ifndef VARIATIONGENERATOR_H_A7DF9144_A921_43B8_9E9A_495433F1A256
#define VARIATIONGENERATOR_H_A7DF9144_A921_43B8_9E9A_495433F1A256

#include <cuda_runtime_api.h>

#include <thrust/count.h>
#include <thrust/host_vector.h>

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

	struct NodeTypeHistogram
	{
		std::vector<unsigned int> typeCounts;

		NodeTypeHistogram()
		{}

		NodeTypeHistogram(const thrust::host_vector<unsigned int>& aTypeArray)
		{
			unsigned int numTypes = 1u + thrust::reduce(aTypeArray.begin(), aTypeArray.end(), 0u, thrust::maximum<unsigned int>());
			typeCounts.resize(numTypes);
			for (unsigned int typeId = 0; typeId < numTypes; ++typeId)
				typeCounts[typeId] = (unsigned int)thrust::count(aTypeArray.begin(), aTypeArray.end(), typeId);
		}

		bool operator ==(const NodeTypeHistogram& aHistogram)
		{
			if (typeCounts.size() != aHistogram.typeCounts.size())
				return false;
			auto myIt = typeCounts.begin();
			auto aIt = aHistogram.typeCounts.begin();
			for (; myIt != typeCounts.end(); ++myIt, ++aIt)
				if (*myIt != *aIt)
					return false;
			return true;
		}
	};

public:
	static const unsigned int MAX_ALLOWED_VARIATIONS = 1000u;
	__host__ Graph mergeGraphs(
		const Graph& aGraph1,
		const Graph& aGraph2,
		const thrust::host_vector<unsigned int>& aSubgraphFlags1,
		const thrust::host_vector<unsigned int>& aSubgraphFlags2,
		const thrust::host_vector<unsigned int>::iterator aSubgraphSeam1Begin,
		const thrust::host_vector<unsigned int>::iterator aSubgraphSeam2Begin,
		const thrust::host_vector<unsigned int>::iterator aSubgraphSeamFlags12Begin,
		const size_t aSeamSize
		);

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
