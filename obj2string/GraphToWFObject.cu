#include "pch.h"
#include "GraphToWFObject.h"

#include "Algebra.h"
#include "WFObjUtils.h"
#include "PartOrientationUtils.h"

#include <thrust/reduce.h>

__host__ WFObject WFObjectGenerator::operator()(
	//example shapes
	WFObject & aObj1,
	WFObject & aObj2,
	//example shape graphs
	Graph & aGraph1,
	Graph & aGraph2,
	//target shape graph
	Graph & aGraph3,
	//estimated edge configurations
	thrust::host_vector<unsigned int>& aEdgeTypes1,
	thrust::host_vector<unsigned int>& aEdgeTypes2,
	thrust::host_vector<unsigned int>& aEdgeTypes3)
{
	PartOrientationEstimator orientations1;
	orientations1.init(aObj1, aGraph1);

	PartOrientationEstimator orientations2;
	orientations2.init(aObj2, aGraph2);

	thrust::host_vector<float3> objCenters1;
	thrust::host_vector<float> objSizes1;

	ObjectCenterExporter()(aObj1, objCenters1, objSizes1);

	thrust::host_vector<float3> objCenters2;
	thrust::host_vector<float> objSizes2;

	ObjectCenterExporter()(aObj2, objCenters2, objSizes2);


	WFObject outputObj;

	unsigned int numNodes = (unsigned)aGraph3.intervals.size() - 1u;
	thrust::host_vector<unsigned int> visited(numNodes, 0u);
	thrust::host_vector<unsigned int> intervalsHost(aGraph3.intervals);
	thrust::host_vector<unsigned int> adjacencyValsHost(aGraph3.adjacencyVals);

	if (seedNodeId >= (unsigned int)numNodes)
	{
		std::default_random_engine generator(seed);
		std::uniform_int_distribution<unsigned int> distribution(0u, (unsigned int)numNodes - 1u);
		seedNodeId = distribution(generator);
	}

	std::deque<unsigned int> frontier;
	frontier.push_back(seedNodeId);
	visited[seedNodeId] = 1u;

	unsigned int seedEdgeId = aGraph3.neighborsBegin(seedNodeId);
	unsigned int seedEdgeType = aEdgeTypes3[seedEdgeId];

	unsigned int seedNodeObj1 = findCorresponingEdge(aGraph1, aEdgeTypes1, seedEdgeType).first;
	unsigned int seedNodeObj2 = findCorresponingEdge(aGraph2, aEdgeTypes2, seedEdgeType).first;

	if (seedNodeObj1 == (unsigned)-1 && seedNodeObj2 == (unsigned)-1)
	{
		std::cerr << "Failed to initialize WFObject creation at node " << seedNodeId << "\n";
		return outputObj;
	}
	if (seedNodeObj1 != (unsigned)-1)
	{
		thrust::host_vector<unsigned int> subgraphFlags1(aObj1.getNumObjects(), 0u);
		subgraphFlags1[seedNodeObj1] = 1u;
		outputObj = insertPieces(outputObj, aObj1, subgraphFlags1, -objCenters1[seedNodeObj1], orientations1.getAbsoluteRotation(seedNodeObj1).conjugate());
	}
	else
	{
		thrust::host_vector<unsigned int> subgraphFlags2(aObj2.getNumObjects(), 0u);
		subgraphFlags2[seedNodeObj2] = 1u;
		outputObj = insertPieces(outputObj, aObj2, subgraphFlags2, -objCenters2[seedNodeObj2], orientations2.getAbsoluteRotation(seedNodeObj2).conjugate());
	}



	//while (!frontier.empty())
	//{
	//	const unsigned int nodeId = frontier.front();
	//	frontier.pop_front();

	//	processNeighbors(
	//		aObj,
	//		nodeId,
	//		visited,
	//		intervalsHost,
	//		adjacencyValsHost,
	//		nodeTypesHost);

	//	for (unsigned int nbrId = intervalsHost[nodeId]; nbrId < intervalsHost[nodeId + 1]; ++nbrId)
	//	{
	//		const unsigned int nodeId = adjacencyValsHost[nbrId];
	//		if (visited[nodeId] == 0u)
	//		{
	//			frontier.push_back(nodeId);
	//			visited[nodeId] = 1u;
	//		}
	//	}
	//}


	return outputObj;
}

__host__ WFObject WFObjectGenerator::insertPieces(
	const WFObject& aObj1,
	const WFObject& aObj2,
	const thrust::host_vector<unsigned int>& subgraphFlags,
	const float3& aTranslation,
	const quaternion4f& aRotation)
{
	thrust::host_vector<unsigned int> subgraphFlags1(aObj1.getNumObjects(), 1u);
	float3 translation1 = make_float3(0.f, 0.f, 0.f);
	float3 translation2 = aTranslation;
	quaternion4f rotation2 = aRotation;
	WFObject result = WFObjectMerger()(aObj1, translation1, aObj2, translation2, rotation2, subgraphFlags1, subgraphFlags);
	return result;
}

__host__ std::pair<unsigned int, unsigned int> WFObjectGenerator::findCorresponingEdge(Graph & aGraph1, thrust::host_vector<unsigned int>& aEdgeTypes1, unsigned int aTargetEdgeType)
{
	std::vector<unsigned int> permutedIds(aEdgeTypes1.size());
	for (unsigned int i = 0u; i < permutedIds.size(); ++i)
		permutedIds[i] = i;
	std::shuffle(permutedIds.begin(), permutedIds.end(), mRNG);

	for (size_t i = 0; i < aEdgeTypes1.size(); ++i)
	{
		size_t edgeId = permutedIds[i];
		if (aEdgeTypes1[edgeId] == aTargetEdgeType)
		{
			return std::make_pair(aGraph1.adjacencyKeys[edgeId], aGraph1.adjacencyVals[edgeId]);
		}
	}
	return std::make_pair((unsigned)-1, (unsigned)-1);
}
