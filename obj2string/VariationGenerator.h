#ifdef _MSC_VER
#pragma once
#endif

#ifndef VARIATIONGENERATOR_H_A7DF9144_A921_43B8_9E9A_495433F1A256
#define VARIATIONGENERATOR_H_A7DF9144_A921_43B8_9E9A_495433F1A256

#include <cuda_runtime_api.h>

#include <thrust/count.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include "WFObject.h"
#include "Graph.h"




class KISSRandomNumberGenerator
{
public:
	uint data[4];
	//data[0],
	//data[1],//must be zero
	//data[2],
	//data[3]; //doesn't need to be re-seeded but must be < 698769069

	__host__ __device__ KISSRandomNumberGenerator(
		const uint aX = 123456789u,
		const uint aY = 362436069u,
		const uint aZ = 521288629u,
		const uint aW = 416191069u)
	{
		data[0] = (aX); data[1] = (aY); data[2] = (aZ); data[3] = (aW);
	}

	__host__ __device__ float operator()()
	{
		data[2] = (36969 * (data[2] & 65535) + (data[2] >> 16)) << 16;
		data[3] = 18000 * (data[3] & 65535) + (data[3] >> 16) & 65535;
		data[0] = 69069 * data[0] + 1234567;
		data[1] = (data[1] = (data[1] = data[1] ^ (data[1] << 17)) ^ (data[1] >> 13)) ^ (data[1] << 5);
		return ((data[2] + data[3]) ^ data[0] + data[1]) * 2.328306E-10f;
	}
};

class HaltonNumberGenerator
{

public:
	const float mPrimesRCP[11] = { 0.5f, 0.333333f, 0.2f, 0.142857f,
		0.09090909f, 0.07692307f, 0.058823529f, 0.0526315789f, 0.04347826f,
		0.034482758f, 0.032258064f };

	__device__ __host__ float operator()(const int aSeed, const int aDimension) const
	{
		if (aDimension < 11)
		{
			float res = 0.f;
			float basisRCP = mPrimesRCP[aDimension];
			const float BASISRCP = mPrimesRCP[aDimension];
			float seed = static_cast<float>(aSeed);

			while (seed)
			{
				float tmp = seed * BASISRCP;
#ifdef __CUDA_ARCH___
				seed = truncf(tmp);
#else
				seed = static_cast<float>(static_cast<int>(tmp));
#endif
				res += basisRCP * (tmp - seed);
				basisRCP *= mPrimesRCP[aDimension];

			}

			return res;
		}
		else
		{
			return 2.f;
		}

	}
};


class VariationGenerator
{
	float totalTime;
	float initTime;
	float samplingTime;
	float matchingTime;
	float svdTime;
	float cpyBackTime;
	float histTime;
	float transformTime;
	float collisionTime;
	float exportTime;
	float conversionTime;
	size_t numVariations;
	size_t histoChecks;
	size_t matchingCuts;
	size_t matchingCutsAndTs;
	size_t histoChecksPassed;

	struct NodeTypeHistogram
	{
		thrust::host_vector<unsigned int> typeCounts;

		__host__ NodeTypeHistogram()
		{}
		
		__host__ NodeTypeHistogram(size_t aNumTypes)
		{
			typeCounts = thrust::host_vector<unsigned int>(aNumTypes, 0u);
		}

		__host__ NodeTypeHistogram(const thrust::host_vector<unsigned int>& aTypeArray)
		{
			unsigned int numTypes = 1u + thrust::reduce(aTypeArray.begin(), aTypeArray.end(), 0u, thrust::maximum<unsigned int>());
			typeCounts.resize(numTypes);
			for (unsigned int typeId = 0; typeId < numTypes; ++typeId)
				typeCounts[typeId] = (unsigned int)thrust::count(aTypeArray.begin(), aTypeArray.end(), typeId);
		}

		__host__ bool operator == (const NodeTypeHistogram& aHistogram) const
		{
			if (typeCounts.size() != aHistogram.typeCounts.size())
				return false;
			for (size_t id = 0; id < typeCounts.size(); ++id)
				if (typeCounts[id] != aHistogram.typeCounts[id])
					return false;
			return true;
		}
	};

public:
	bool writeVariations; //write out generated variations in .obj 
	bool writeVariationGraphs; //write out the graphs of the generated variations
	bool multiString;  //return single or multiple SMILES strings per variation
	bool fixVariation; //try to repair relative part orientation in the variation
	__host__ VariationGenerator() : 
		writeVariations(false), 
		writeVariationGraphs(true), 
		multiString(true),
		fixVariation(false)
	{}

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
