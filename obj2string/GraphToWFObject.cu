#include "pch.h"
#include "GraphToWFObject.h"

#include "Algebra.h"
#include "WFObjUtils.h"
#include "Graph2String.h"
#include "CollisionDetector.h"
#include "Wiggle.h"
#include "DebugUtils.h"

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
	unsigned int numEdgeTypes = 1u + thrust::reduce(aEdgeTypes1.begin(), aEdgeTypes1.end(), 0u, thrust::maximum<unsigned int>());

	mOrientations1.init(aObj1, aGraph1);
	mOrientations2.init(aObj2, aGraph2);
	
	thrust::host_vector<float> objSizes1;
	ObjectCenterExporter()(aObj1, objCenters1, objSizes1);
	
	thrust::host_vector<float> objSizes2;
	ObjectCenterExporter()(aObj2, objCenters2, objSizes2);

	GrammarCheck grammarCheck;
	grammarCheck.init(aObj1, aGraph1.intervals, aGraph1.adjacencyVals);
	grammarCheck.init(aObj2, aGraph2.intervals, aGraph2.adjacencyVals);

	CollisionDetector detector;

	Wiggle wiggle;
	wiggle.init(aObj1, aGraph1);
	wiggle.init(aObj2, aGraph2);

	WFObject outputObj;

	unsigned int numNodes = (unsigned)aGraph3.numNodes();
	thrust::host_vector<unsigned int> visited(numNodes, 0u);
	thrust::host_vector<unsigned int> intervalsHost(aGraph3.intervals);
	thrust::host_vector<unsigned int> adjacencyKeysHost(aGraph3.adjacencyKeys);
	thrust::host_vector<unsigned int> adjacencyValsHost(aGraph3.adjacencyVals);

	std::uniform_int_distribution<unsigned int> randNodeId(0u, numNodes - 1u);
	seedNodeId = randNodeId(mRNG);

	std::uniform_int_distribution<unsigned int> randEdgeId(intervalsHost[seedNodeId], intervalsHost[seedNodeId + 1] - 1);

	unsigned int seedEdgeId = randEdgeId(mRNG);
	unsigned int seedEdgeType = aEdgeTypes3[seedEdgeId];
	unsigned int reverseSeedEdgeId = getOpositeEdgeId(seedEdgeId, intervalsHost, adjacencyKeysHost, adjacencyValsHost);
	unsigned int reverseTypeId = aEdgeTypes3[reverseSeedEdgeId];


	unsigned int correspondingEdgeG1 = findCorresponingEdgeId(aGraph1, aEdgeTypes1, seedEdgeType, reverseTypeId);
	unsigned int correspondingEdgeG2 = findCorresponingEdgeId(aGraph2, aEdgeTypes2, seedEdgeType, reverseTypeId);
	
	if (correspondingEdgeG1 >= (unsigned)aGraph1.adjacencyKeys.size() && correspondingEdgeG2 >= (unsigned)aGraph2.adjacencyKeys.size())
	{
		//std::cerr << "Failed to initialize WFObject creation at node " << seedNodeId << "\n";
		return outputObj;
	}
	if (correspondingEdgeG1 < aGraph1.adjacencyKeys.size())
	{
		unsigned int seedNodeObj1 = aGraph1.adjacencyKeys[correspondingEdgeG1];
		thrust::host_vector<unsigned int> subgraphFlags1(aObj1.getNumObjects(), 0u);
		subgraphFlags1[seedNodeObj1] = 1u;
		float3 zero = make_float3(0.f, 0.f, 0.f);
		outputObj = insertPieces(outputObj, aObj1, subgraphFlags1, zero, objCenters1[seedNodeObj1], make_quaternion4f(0.f,0.f,0.f,1.f));
	}
	else
	{
		unsigned int seedNodeObj2 = aGraph2.adjacencyKeys[correspondingEdgeG2];
		thrust::host_vector<unsigned int> subgraphFlags2(aObj2.getNumObjects(), 0u);
		subgraphFlags2[seedNodeObj2] = 1u;
		float3 zero = make_float3(0.f, 0.f, 0.f);
		outputObj = insertPieces(outputObj, aObj2, subgraphFlags2, zero, objCenters2[seedNodeObj2], make_quaternion4f(0.f, 0.f, 0.f, 1.f));
	}

	std::deque<unsigned int> frontier;
	frontier.push_back(seedNodeId);
	visited[seedNodeId] = 1u;

	unsigned int insertedNodeCount = 0u;
	thrust::host_vector<unsigned int> nodeIdMap(aGraph3.numNodes(), 0u);
	nodeIdMap[seedNodeId] = insertedNodeCount++;

	//std::string dbgFileName("embed_test_");
	//unsigned int numIterations = 0u;

	while (!frontier.empty())
	{
		const unsigned int nodeId = frontier.front();
		frontier.pop_front();
		
		if (nodeIdMap[nodeId] >= outputObj.getNumObjects())
		{
			std::cerr << "Trying to use non-existing obj-object id " << nodeIdMap[nodeId] << "\n";
			std::cerr << "Max obj-object id " << outputObj.getNumObjects() << "\n";
			std::cerr << "Number of insertions: " << insertedNodeCount << "\n";
			continue;
		}
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

		////////////////////////////////////////////////////////////////////////////////////////////
		//pairwise neighbor configurations
		std::vector<PartOrientationEstimator::PariwiseNeighborConfiguration> neigborConfiguration;
		size_t faceId = outputObj.objects[nodeIdMap[nodeId]].x;
		size_t materialId = outputObj.faces[faceId].material;
		unsigned int typeA = (unsigned int)materialId;
		const float sizeOfNode = outputObj.getObjectSize(nodeIdMap[nodeId]);

		for (unsigned int nbrId = intervalsHost[nodeId]; nbrId < intervalsHost[nodeId + 1]; ++nbrId)
		{
			const unsigned int neighborId = adjacencyValsHost[nbrId];
			if (visited[neighborId] == 1u)
			{
				size_t faceId = outputObj.objects[nodeIdMap[neighborId]].x;
				size_t materialId = outputObj.faces[faceId].material;
				unsigned int typeNbrB1 = (unsigned int)materialId;
				float3 nbrB1Center = outputObj.getObjectCenter(insertedNodeCount);

				for (unsigned int nbrId1 = intervalsHost[nodeId]; nbrId1 < nbrId; ++nbrId1)
				{
					const unsigned int neighborId1 = adjacencyValsHost[nbrId1];
					if (visited[neighborId1] == 1u)
					{
						size_t faceId = outputObj.objects[nodeIdMap[neighborId1]].x;
						size_t materialId = outputObj.faces[faceId].material;
						unsigned int typeNbrB0 = (unsigned int)materialId;
						float3 nbrB0Center = outputObj.getObjectCenter(nodeIdMap[neighborId1]);
						PartOrientationEstimator::PariwiseNeighborConfiguration current;
						current.typeA = typeA;
						current.typeNbrB0 = typeNbrB0;
						current.typeNbrB1 = typeNbrB1;
						current.dist = len(nbrB1Center - nbrB0Center);
						current.size = sizeOfNode;
						neigborConfiguration.push_back(current);
					}
				}
			}
		}
		////////////////////////////////////////////////////////////////////////////////////////////

		for (unsigned int nbrId = intervalsHost[nodeId]; nbrId < intervalsHost[nodeId + 1]; ++nbrId)
		{
			const unsigned int neighborId = adjacencyValsHost[nbrId];
			unsigned int typeB = (unsigned)-1;
			if (visited[neighborId] == 0u)
			{

				unsigned int currentEdgeId = nbrId;
				unsigned int currentEdgeType = aEdgeTypes3[currentEdgeId] < numEdgeTypes ? aEdgeTypes3[currentEdgeId] : (unsigned)-1;

				unsigned int reverseEdgeId = getOpositeEdgeId(currentEdgeId, intervalsHost, adjacencyKeysHost, adjacencyValsHost);
				unsigned int reverseEdgeType = aEdgeTypes3[reverseEdgeId] < numEdgeTypes ? aEdgeTypes3[reverseEdgeId] : (unsigned)-1;

				typeB = getNodeType(aGraph1, aObj1, aEdgeTypes1, currentEdgeType, typeA);
				if(typeB == (unsigned)-1)
					typeB = getNodeType(aGraph1, aObj1, aEdgeTypes1, reverseEdgeType, typeA);

				for (size_t numAttempts = 0u; numAttempts < 24; ++numAttempts)
				{

					unsigned int correspondingEdgeIdObj1 = findCorresponingEdgeId(aGraph1, aEdgeTypes1, currentEdgeType, reverseEdgeType);
					unsigned int correspondingEdgeIdObj2 = findCorresponingEdgeId(aGraph2, aEdgeTypes2, currentEdgeType, reverseEdgeType);

					if ((numAttempts >= 12 || currentEdgeType == (unsigned)-1) && typeB != (unsigned)-1)
					{
						//assume wrong requested edge type
						//try a random edge with matching node types
						correspondingEdgeIdObj1 = findRandomEdgeId(aGraph1, aObj1, typeA, typeB);
						correspondingEdgeIdObj2 = findRandomEdgeId(aGraph2, aObj2, typeA, typeB);
					}

					std::vector<PartOrientationEstimator::PariwiseNeighborConfiguration> currentConfiguration(neigborConfiguration);

					WFObject tmpObj;
					if (correspondingEdgeIdObj2 != (unsigned)-1 && (randNodeId(mRNG) < numNodes / 2u || correspondingEdgeIdObj1 == (unsigned)-1))
					{
						tmpObj = appendNode(outputObj, correspondingEdgeIdObj2, aGraph2, aObj2, objCenters2, mOrientations2, rotationA, translationA);								
					}
					else if (correspondingEdgeIdObj1 != (unsigned)-1)
					{
						tmpObj = appendNode(outputObj, correspondingEdgeIdObj1, aGraph1, aObj1, objCenters1, mOrientations1, rotationA, translationA);
					}
					else
					{
						//std::cerr << "Skipping WFObject node " << neighborId << "\n";
						//std::cerr << "(After inserting " << insertedNodeCount << " nodes.)\n";
						//std::cerr << "Edge type A->B requested: " << currentEdgeType << "\n";
						//std::cerr << "Edge type B->A requested: " << reverseEdgeType << "\n";
						//std::cerr << "Edge id : " << currentEdgeId << " out of " << aEdgeTypes3.size() << "\n";
						//std::cerr << "Reverse edge id : " << reverseEdgeId << " out of " << aEdgeTypes3.size() << "\n";
						continue;
					}

					Graph testGraph = detector.computeCollisionGraph(tmpObj, 0.0f);
					if (!grammarCheck.checkSubgraph(tmpObj, testGraph.intervals, testGraph.adjacencyVals))
					{
						//WFObjectFileExporter()(tmpObj, (dbgFileName + itoa(numIterations++)).c_str());
						continue;
					}

					if (!wiggle.fixEdge(tmpObj, nodeIdMap[nodeId], insertedNodeCount, 64))
					{
						continue;
					}

					////////////////////////////////////////////////////////////////////////////////////////////
					//pairwise neighbor configurations
					unsigned int typeNbrB1 = (unsigned int)materialId;
					float3 nbrB1Center = tmpObj.getObjectCenter(insertedNodeCount);

					for (unsigned int nbrId1 = intervalsHost[nodeId]; nbrId1 < nbrId; ++nbrId1)
					{
						const unsigned int neighborId1 = adjacencyValsHost[nbrId1];
						if (visited[neighborId1] == 1u)
						{
							size_t faceId = tmpObj.objects[nodeIdMap[neighborId1]].x;
							size_t materialId = tmpObj.faces[faceId].material;
							unsigned int typeNbrB0 = (unsigned int)materialId;
							float3 nbrB0Center = tmpObj.getObjectCenter(nodeIdMap[neighborId1]);
							PartOrientationEstimator::PariwiseNeighborConfiguration current;
							current.typeA = typeA;
							current.typeNbrB0 = typeNbrB0;
							current.typeNbrB1 = typeNbrB1;
							current.dist = len(nbrB1Center - nbrB0Center);
							current.size = sizeOfNode;
							currentConfiguration.push_back(current);
						}
					}
					if (numAttempts < 16 &&
						!mOrientations1.checkNeighborConfiguration(currentConfiguration)
						&& !mOrientations2.checkNeighborConfiguration(currentConfiguration))
					{
						continue;
					}
					else
					{
						neigborConfiguration = currentConfiguration;
					}
					////////////////////////////////////////////////////////////////////////////////////////////
				
					outputObj = tmpObj;

					nodeIdMap[neighborId] = insertedNodeCount++;
					frontier.push_back(neighborId);
					visited[neighborId] = 1u;
					break;
				}//end for number of node embedding attempts

			}//end if visited neighbor
		}
		//break;
		//WFObjectFileExporter()(outputObj, (dbgFileName + itoa(numIterations++)).c_str());
	}


	return outputObj;
}

WFObject WFObjectGenerator::appendNode(
	WFObject &outputObj,
	unsigned int correspondingEdgeIdObj1,
	Graph & aGraph,
	WFObject & aObj,
	thrust::host_vector<float3>& aCenters,
	PartOrientationEstimator& aOrientations,
	const quaternion4f &rotationA,
	const float3 &translationA)
{
	unsigned int correspondingNodeIdObj = aGraph.adjacencyKeys[correspondingEdgeIdObj1];
	unsigned int correspondingNeighborIdObj = aGraph.adjacencyVals[correspondingEdgeIdObj1];

	float3 translationA1 = aCenters[correspondingNodeIdObj];
	quaternion4f rotationA1 = aOrientations.getAbsoluteRotation(correspondingNodeIdObj);

	quaternion4f relativeR = rotationA * rotationA1.conjugate();
	if (isIdentity(relativeR, 0.001f))
		relativeR = make_quaternion4f(0.f, 0.f, 0.f, 1.f);

	thrust::host_vector<unsigned int> subgraphFlags(aObj.getNumObjects(), 0u);
	subgraphFlags[correspondingNeighborIdObj] = 1u;
	return insertPieces(outputObj, aObj, subgraphFlags, translationA, translationA1, relativeR);
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
	WFObject result = WFObjectMerger()(aObj1, translation1, aObj2, translation2, rotation2, subgraphFlags1, aSubgraphFlags2, false);
	return result;
}

__host__ unsigned int WFObjectGenerator::findCorresponingEdgeId(
	Graph& aGraph,
	thrust::host_vector<unsigned int>& aEdgeTypes1,
	unsigned int aTargetEdgeType,
	unsigned int aTargetReverseType)
{
	std::vector<unsigned int> permutedIds(aEdgeTypes1.size());
	for (unsigned int i = 0u; i < permutedIds.size(); ++i)
		permutedIds[i] = i;
	std::shuffle(permutedIds.begin(), permutedIds.end(), mRNG);

	unsigned int bestInvalidEdgeId = (unsigned)-1;
	for (size_t i = 0; i < aEdgeTypes1.size(); ++i)
	{
		unsigned int edgeId = permutedIds[i];
		unsigned int reverseId = aGraph.getOpositeEdgeId(edgeId);
		if (aEdgeTypes1[edgeId] == aTargetEdgeType && aEdgeTypes1[reverseId] == aTargetReverseType)
		{
			return edgeId;
		}
		else if (aEdgeTypes1[edgeId] == aTargetReverseType && aEdgeTypes1[reverseId] == aTargetEdgeType)
		{
			return reverseId;
		}
		else if ((!strictEmbeddingFlag || aTargetReverseType == (unsigned)-1)  && aEdgeTypes1[edgeId] == aTargetEdgeType)
		{
			bestInvalidEdgeId = edgeId;
		}
		else if ((!strictEmbeddingFlag || aTargetReverseType == (unsigned)-1) && aEdgeTypes1[reverseId] == aTargetEdgeType)
		{
			bestInvalidEdgeId = reverseId;
		}
	}

	return bestInvalidEdgeId;
}

__host__ unsigned int WFObjectGenerator::findRandomEdgeId(
	Graph & aGraph,
	WFObject & aObj,
	unsigned int aNodeType0,
	unsigned int aNodeType1)
{
	std::vector<unsigned int> permutedIds(aObj.getNumObjects());
	for (unsigned int i = 0u; i < permutedIds.size(); ++i)
		permutedIds[i] = i;
	std::shuffle(permutedIds.begin(), permutedIds.end(), mRNG);

	thrust::host_vector<unsigned int> intervalsHost(aGraph.intervals);
	thrust::host_vector<unsigned int> adjacencyValsHost(aGraph.adjacencyVals);

	for (size_t i = 0; i < permutedIds.size(); ++i)
	{
		unsigned int nodeId0 = permutedIds[i];
		size_t faceId = aObj.objects[nodeId0].x;
		size_t materialId = aObj.faces[faceId].material;
		unsigned int type0 = (unsigned int)materialId;

		if (type0 != aNodeType0)
			continue;

		for (unsigned int edgeId = intervalsHost[nodeId0]; edgeId < intervalsHost[nodeId0 + 1]; ++edgeId)
		{
			unsigned int nodeId1 = adjacencyValsHost[edgeId];
			size_t faceId = aObj.objects[nodeId1].x;
			size_t materialId = aObj.faces[faceId].material;
			unsigned int type1 = (unsigned int)materialId;
			if (type1 == aNodeType1)
				return edgeId;
		}
	}

	return (unsigned) -1;
}

__host__ unsigned int WFObjectGenerator::getNodeType(
	Graph& aGraph,
	WFObject& aObj,
	thrust::host_vector<unsigned int>& aEdgeTypes,
	unsigned int aEdgeType,
	unsigned int aNbrType)
{
	for (unsigned int edgeId = 0u; edgeId < aEdgeTypes.size(); ++edgeId)
	{
		if (aEdgeTypes[edgeId] != aEdgeType)
			continue;

		unsigned int nodeIdA = aGraph.adjacencyKeys[edgeId];
		size_t faceIdA = aObj.objects[nodeIdA].x;
		unsigned int typeA = (unsigned int)aObj.faces[faceIdA].material;

		unsigned int nodeIdB = aGraph.adjacencyVals[edgeId];
		size_t faceIdB = aObj.objects[nodeIdB].x;
		unsigned int typeB = (unsigned int)aObj.faces[faceIdB].material;

		if (typeA == aNbrType)
			return typeB;
		else if(typeB == aNbrType)
			return typeA;
	}
	return (unsigned int)-1;
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
