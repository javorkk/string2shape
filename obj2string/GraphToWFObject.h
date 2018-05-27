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

	__host__ WFObjectGenerator()
	{
		seed = (unsigned int)std::chrono::system_clock::now().time_since_epoch().count();
		seedNodeId = (unsigned)-1;
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
		Graph & aGraph1,
		thrust::host_vector<unsigned int>& aEdgeTypes1,
		unsigned int aTargetEdgeType);

	__host__ void translateObj(
		WFObject& aObj,
		unsigned int aObjId,
		const float3 & aTranslation);
};


#endif //GRAPHTOWFOBJECT_H_5D48161E_399D_4137_9C2A_70959D186A94
