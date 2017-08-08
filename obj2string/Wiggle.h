#ifdef _MSC_VER
#pragma once
#endif

#ifndef WIGGLE_H_D31D930E_4861_4383_901F_3FC7AA0D14AC
#define WIGGLE_H_D31D930E_4861_4383_901F_3FC7AA0D14AC

#include <cuda_runtime_api.h>
#include <vector>
#include <set>
#include <string>

#include "Algebra.h"
#include "WFObject.h"
#include "Graph.h"

#include <thrust/host_vector.h>

class Wiggle
{
	thrust::host_vector<unsigned int> mIntervals;
	//Node a's type
	thrust::host_vector<unsigned int> mNeighborTypeKeys;
	//Node b's type
	thrust::host_vector<unsigned int> mNeighborTypeVals;
	//Center of gravity of b in a's coordinate system
	thrust::host_vector<float3> mRelativeTranslation;
	//Rotates a's local coordinate frame into b's
	thrust::host_vector<quaternion4f> mRelativeRotation;

public:
	__host__ void init(
		WFObject& aObj,
		Graph & aGraph);

	__host__ void refine(
		WFObject& aObj,
		Graph & aGraph);

	__host__ void depthFirstTraverse(
		WFObject& aObj,
		unsigned int v,
		thrust::host_vector<unsigned int>& visited,
		unsigned int parent,
		thrust::host_vector<unsigned int>& intervalsHost,
		thrust::host_vector<unsigned int>& adjacencyValsHost,
		thrust::host_vector<unsigned int>& nodeTypeIds);
};


#endif // WIGGLE_H_D31D930E_4861_4383_901F_3FC7AA0D14AC
