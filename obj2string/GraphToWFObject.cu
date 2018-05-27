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

	mOrientations1.init(aObj1, aGraph1);
	mOrientations2.init(aObj2, aGraph2);
	
	thrust::host_vector<float> objSizes1;
	ObjectCenterExporter()(aObj1, objCenters1, objSizes1);
	
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

	unsigned int seedEdgeId = aGraph3.neighborsBegin(seedNodeId);
	unsigned int seedEdgeType = aEdgeTypes3[seedEdgeId];

	unsigned int seedNodeObj1 = aGraph1.adjacencyKeys[findCorresponingEdgeId(aGraph1, aEdgeTypes1, seedEdgeType)];
	unsigned int seedNodeObj2 = aGraph2.adjacencyKeys[findCorresponingEdgeId(aGraph2, aEdgeTypes2, seedEdgeType)];

	if (seedNodeObj1 == (unsigned)-1 && seedNodeObj2 == (unsigned)-1)
	{
		std::cerr << "Failed to initialize WFObject creation at node " << seedNodeId << "\n";
		return outputObj;
	}
	if (seedNodeObj1 != (unsigned)-1)
	{
		thrust::host_vector<unsigned int> subgraphFlags1(aObj1.getNumObjects(), 0u);
		subgraphFlags1[seedNodeObj1] = 1u;
		float3 zero = make_float3(0.f, 0.f, 0.f);
		outputObj = insertPieces(outputObj, aObj1, subgraphFlags1, zero, -objCenters1[seedNodeObj1], make_quaternion4f(0.f,0.f,0.f,1.f));
	}
	else
	{
		thrust::host_vector<unsigned int> subgraphFlags2(aObj2.getNumObjects(), 0u);
		subgraphFlags2[seedNodeObj2] = 1u;
		float3 zero = make_float3(0.f, 0.f, 0.f);
		outputObj = insertPieces(outputObj, aObj2, subgraphFlags2, zero, -objCenters2[seedNodeObj2], make_quaternion4f(0.f, 0.f, 0.f, 1.f));
	}

	std::deque<unsigned int> frontier;
	frontier.push_back(seedNodeId);
	visited[seedNodeId] = 1u;

	unsigned int indertedNodeCount = 0u;
	thrust::host_vector<unsigned int> nodeIdMap(aGraph3.numNodes(), 0u);
	nodeIdMap[seedNodeId] = indertedNodeCount++;
	
	while (!frontier.empty())
	{
		const unsigned int nodeId = frontier.front();
		frontier.pop_front();
		
		const unsigned int nodeCount = 1u;
		thrust::host_vector<unsigned int> nodeIds(1, nodeIdMap[nodeId]);

		thrust::host_vector<float3> vertexBufferHost;
		thrust::host_vector<uint2> vtxRanges;

		VertexBufferUnpacker unpackVertices;
		unpackVertices(outputObj, nodeIds, vtxRanges, vertexBufferHost);

		//Use PCA to compute local coordiante system for each object
		thrust::host_vector<float3> translations(1);
		thrust::host_vector<quaternion4f> rotations(1);
		thrust::host_vector<double> tmpCovMatrix(1 * 3 * 3, 0.f);
		thrust::host_vector<double> tmpDiagonalW(1 * 3);
		thrust::host_vector<double> tmpMatrixV(1 * 3 * 3);
		thrust::host_vector<double> tmpVecRV(1 * 3);

		LocalCoordsEstimator estimateT(
			thrust::raw_pointer_cast(vtxRanges.data()),
			thrust::raw_pointer_cast(vertexBufferHost.data()),
			thrust::raw_pointer_cast(tmpCovMatrix.data()),
			thrust::raw_pointer_cast(tmpDiagonalW.data()),
			thrust::raw_pointer_cast(tmpMatrixV.data()),
			thrust::raw_pointer_cast(tmpVecRV.data()),
			thrust::raw_pointer_cast(translations.data()),
			thrust::raw_pointer_cast(rotations.data())
		);

		estimateT(0);

		float3 translationA = translations[0];
		quaternion4f rotationA = rotations[0];

		for (unsigned int nbrId = intervalsHost[nodeId]; nbrId < intervalsHost[nodeId + 1]; ++nbrId)
		{
			const unsigned int neighborId = adjacencyValsHost[nbrId];
			if (visited[neighborId] == 0u)
			{
				frontier.push_back(neighborId);
				visited[neighborId] = 1u;

				unsigned int currentEdgeId = aGraph3.neighborsBegin(nodeId) + nbrId;
				unsigned int currentEdgeType = aEdgeTypes3[currentEdgeId];

				unsigned int correspondingEdgeIdObj1 = findCorresponingEdgeId(aGraph1, aEdgeTypes1, currentEdgeType);

				if (correspondingEdgeIdObj1 == (unsigned)-1)
				{
					std::cerr << "Failed to create WFObject node " << correspondingEdgeIdObj1 << "\n";
					std::cerr << "(After inserting " << indertedNodeCount << " nodes.)\n";

					return outputObj;
				}

				unsigned int correspondingNodeIdObj1 = aGraph1.adjacencyKeys[correspondingEdgeIdObj1];
				
				float3 translationA1 = objCenters1[correspondingNodeIdObj1];
				quaternion4f rotationA1 = mOrientations1.getAbsoluteRotation(correspondingNodeIdObj1);

				quaternion4f relativeR = rotationA * rotationA1.conjugate();
				if (isIdentity(relativeR, 0.001f))
					relativeR = make_quaternion4f(0.f, 0.f, 0.f, 1.f);

				thrust::host_vector<unsigned int> subgraphFlags1(aObj1.getNumObjects(), 0u);
				subgraphFlags1[correspondingNodeIdObj1] = 1u;
				outputObj = insertPieces(outputObj, aObj1, subgraphFlags1, translationA, translationA1, relativeR);

				nodeIdMap[neighborId] = indertedNodeCount++;
			}
		}
	}


	return outputObj;
}

__host__ WFObject WFObjectGenerator::insertPieces(
	const WFObject& aObj1,
	const WFObject& aObj2,
	const thrust::host_vector<unsigned int>& aSubgraphFlags2,
	const float3& aTranslation1,
	const float3& aTranslation2,
	const quaternion4f& aRotation)
{
	thrust::host_vector<unsigned int> subgraphFlags1(aObj1.getNumObjects(), 1u);
	float3 translation1 = aTranslation1;
	float3 translation2 = aTranslation2;
	quaternion4f rotation2 = aRotation;
	WFObject result = WFObjectMerger()(aObj1, translation1, aObj2, translation2, rotation2, subgraphFlags1, aSubgraphFlags2);
	return result;
}

__host__ unsigned int WFObjectGenerator::findCorresponingEdgeId(Graph & aGraph1, thrust::host_vector<unsigned int>& aEdgeTypes1, unsigned int aTargetEdgeType)
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
			return edgeId;
		}
	}
	return (unsigned)-1;
}

__host__ void WFObjectGenerator::translateObj(WFObject & aObj, unsigned int aObjId, const float3 & aTranslation)
{
	thrust::host_vector<unsigned int> processed(aObj.getNumVertices(), 0u);
	for (int faceId = aObj.objects[aObjId].x; faceId < aObj.objects[aObjId].y; ++faceId)
	{
		WFObject::Face face = aObj.faces[faceId];
		size_t vtxId1 = aObj.faces[faceId].vert1;
		size_t vtxId2 = aObj.faces[faceId].vert2;
		size_t vtxId3 = aObj.faces[faceId].vert3;
		if (processed[vtxId1] == 0u)
		{
			processed[vtxId1] = 1u;
			float3 vtx = aObj.vertices[vtxId1];
			aObj.vertices[vtxId1] = vtx + aTranslation;
		}
		if (processed[vtxId2] == 0u)
		{
			processed[vtxId2] = 1u;
			float3 vtx = aObj.vertices[vtxId2];
			aObj.vertices[vtxId2] = vtx + aTranslation;

		}
		if (processed[vtxId3] == 0u)
		{
			processed[vtxId3] = 1u;
			float3 vtx = aObj.vertices[vtxId3];
			aObj.vertices[vtxId3] = vtx + aTranslation;
		}
	}
}
