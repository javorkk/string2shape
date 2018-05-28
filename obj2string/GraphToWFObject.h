#ifdef _MSC_VER
#pragma once
#endif

#ifndef GRAPHTOWFOBJECT_H_5D48161E_399D_4137_9C2A_70959D186A94
#define GRAPHTOWFOBJECT_H_5D48161E_399D_4137_9C2A_70959D186A94

#include "Algebra.h"
#include "WFObject.h"
#include "Graph.h"
#include "PartOrientationUtils.h"

#include <thrust/host_vector.h>

class WFObjectGenerator
{
	std::default_random_engine mRNG;
	PartOrientationEstimator mOrientations1;
	PartOrientationEstimator mOrientations2;
	thrust::host_vector<float3> objCenters1;
	thrust::host_vector<float3> objCenters2;
public:
	unsigned int seed;
	unsigned int seedNodeId;
	bool strictEmbeddingFlag;

	__host__ WFObjectGenerator()
	{
		seed = (unsigned int)std::chrono::system_clock::now().time_since_epoch().count();
		mRNG = std::default_random_engine(seed);
		seedNodeId = (unsigned)-1;
		strictEmbeddingFlag = true;
	}

	__host__ WFObject operator()(
		//example shapes
		WFObject& aObj1,
		WFObject& aObj2,
		//example shape graphs
		Graph& aGraph1,
		Graph& aGraph2,
		//target shape graph
		Graph& aGraph3,
		//estimated edge configurations
		thrust::host_vector<unsigned int>& aEdgeTypes1, 
		thrust::host_vector<unsigned int>& aEdgeTypes2,
		thrust::host_vector<unsigned int>& aEdgeTypes3
		);

	void appendNode(
		WFObject &outputObj,
		unsigned int correspondingEdgeIdObj1,
		Graph & aGraph1,
		WFObject & aObj1,
		const quaternion4f &rotationA,
		const float3 &translationA);

	//inserts Obj-objects from aObj2 into aObj1
	//all Obj-objects in aObj1 participate
	//only flagged Obj-objects in aObj2 participate
	__host__ WFObject insertPieces(
		const WFObject& aObj1,
		const WFObject& aObj2,
		const thrust::host_vector<unsigned int>& aSubgraphFlags2,
		const float3& aTranslation1,
		const float3& aTranslation2,
		const quaternion4f& aRotation);

	__host__ unsigned int findCorresponingEdgeId(
		Graph& aGraph,
		thrust::host_vector<unsigned int>& aEdgeTypes1,
		unsigned int aTargetEdgeType,
		unsigned int aTargetReverseType);

	__host__ void translateObj(
		WFObject& aObj,
		unsigned int aObjId,
		const float3 & aTranslation);

	__host__ FORCE_INLINE unsigned int getOpositeEdgeId(
		unsigned int aEdgeId,
		thrust::host_vector<unsigned int>& intervals,
		thrust::host_vector<unsigned int>& adjacencyKeys,
		thrust::host_vector<unsigned int>& adjacencyVals)
	{
		if (aEdgeId >= adjacencyVals.size())
			return (unsigned)adjacencyVals.size();
		unsigned int oposingKey = adjacencyVals[aEdgeId];
		unsigned int oposingVal = adjacencyKeys[aEdgeId];
		for (unsigned int id = intervals[oposingKey]; id < intervals[oposingKey + 1]; ++id)
			if (adjacencyVals[id] == oposingVal)
				return id;
		return (unsigned)adjacencyVals.size();
	}


};


#endif //GRAPHTOWFOBJECT_H_5D48161E_399D_4137_9C2A_70959D186A94
