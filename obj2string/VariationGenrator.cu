#include "pch.h"
#include "VariationGenerator.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <thrust/scan.h>

#include "Algebra.h"
#include "SVD.h"
#include "WFObjUtils.h"
#include "Graph2String.h"
#include "CollisionDetector.h"
#include "CollisionGraphExporter.h"

#include "DebugUtils.h"
#include "Timer.h"


class DistanceMatrixWriter
{
public:
	size_t stride;
	thrust::device_ptr<float> matrix;
	thrust::device_ptr<float3> positions;


	DistanceMatrixWriter(
		size_t aStride,
		thrust::device_ptr<float> aMatrix,
		thrust::device_ptr<float3> aPositions
	) :stride(aStride), matrix(aMatrix), positions(aPositions)
	{}

	__host__ __device__	void operator()(const size_t& aId)
	{
		const size_t myRowId = aId % stride;
		const size_t myColId = aId / stride;
		const float3 objCenter1 = positions[myColId];
		const float3 objCenter2 = positions[myRowId];
		matrix[myColId + myRowId * stride] = len(objCenter1 - objCenter2);
	}

};

class SubgraphInitializer
{
public:
	unsigned int graphSize;
	unsigned int subgraphSize;
	unsigned int numSubgraphs;
	unsigned int subgraphsPerSeedNode;
	//graph edges
	thrust::device_ptr<unsigned int> adjIntervals;
	thrust::device_ptr<unsigned int> neighborIds;

	thrust::device_ptr<unsigned int> outNodeIds;
	thrust::device_ptr<unsigned int> outBorderNodeFlags;

	SubgraphInitializer(
		unsigned int aGraphSize,
		unsigned int aSampleSize,
		unsigned int aNumSamples,
		thrust::device_ptr<unsigned int> aIntervals,
		thrust::device_ptr<unsigned int> aNeighborIds,
		thrust::device_ptr<unsigned int> outIds,
		thrust::device_ptr<unsigned int> outFlags
		) : graphSize(aGraphSize),
		subgraphSize(aSampleSize),
		numSubgraphs(aNumSamples),
		subgraphsPerSeedNode(aNumSamples / aGraphSize),
		adjIntervals(aIntervals),
		neighborIds(aNeighborIds),
		outNodeIds(outIds),
		outBorderNodeFlags(outFlags)
	{}

	__host__ __device__	void operator()(const size_t& aId_s)
	{
		unsigned int aId = (unsigned int)aId_s;

		KISSRandomNumberGenerator genRand(
			3643u + aId * 4154207u * subgraphsPerSeedNode + numSubgraphs,
			1761919u + aId * 2746753u * subgraphsPerSeedNode ,
			331801u + aId,
			10499029u);

		unsigned int subgraphSeedNodeId = subgraphsPerSeedNode == 0u ? aId : aId / subgraphsPerSeedNode;
		//unsigned int subgraphSeedNodeId = max(min((unsigned int)(genRand() * (float)graphSize), graphSize - 1u), 0u);
		unsigned int subgraphStartLocation = aId * subgraphSize;

		outNodeIds[subgraphStartLocation] = subgraphSeedNodeId;

		unsigned int currentSize = 1u;
		//unsigned int globalNbrCount = 0u;

		//compute subgraph
		for(unsigned int currentDepth = 0u; currentDepth < subgraphSize && currentSize < subgraphSize && currentSize < graphSize; ++currentDepth)
		{
			unsigned int neighborCount = 0u;
			for (unsigned int localNodeId = 0u; localNodeId < currentSize; ++localNodeId)
			{
				unsigned int nodeId = outNodeIds[subgraphStartLocation + localNodeId];
				//const float numNbrsRCP = 1.f / (float)(adjIntervals[nodeId + 1u] - adjIntervals[nodeId]);
				for (unsigned int localNeighborId = adjIntervals[nodeId]; localNeighborId < adjIntervals[nodeId + 1u]; ++localNeighborId)
				{
					unsigned int neighborId = neighborIds[localNeighborId];					
					bool alreadyIncluded = false;
					for (unsigned int previousNodeId = 0u; previousNodeId < currentSize + neighborCount; ++previousNodeId)
					{						
						if (outNodeIds[subgraphStartLocation + previousNodeId] == neighborId)
						{
							alreadyIncluded = true;
							break;
						}
					}

					//if (!alreadyIncluded && subgraphsPerSeedNode > 0u)
					//{
					//	//deterministically discard the neighbor
					//	unsigned int subgraphSampleId = aId % subgraphsPerSeedNode;
					//	unsigned int includeFlag = (subgraphSampleId >> globalNbrCount) & 0x00000001;
					//	if (includeFlag != 0u)
					//	{							
					//		alreadyIncluded = true; 
					//	}
					//	++globalNbrCount;
					//}

					if (!alreadyIncluded)
					{
						//randomly discard the neighbor node
						alreadyIncluded = genRand() < 0.5f; // numNbrsRCP;
					}

					if (!alreadyIncluded && neighborCount + currentSize < subgraphSize) //add to subgraph
					{
						outNodeIds[subgraphStartLocation + neighborCount + currentSize] = neighborId;
						neighborCount++;
					}
					else if (!alreadyIncluded && neighborCount > 0u  && genRand() < 0.5)//replace a random node with the same depth
					{
						unsigned int randLocation = (int)(genRand() * (float)(neighborCount));
						outNodeIds[subgraphStartLocation + randLocation + currentSize] = neighborId;
					}
				}

			}

			currentSize += neighborCount;
		}

		//compute subgraph interior and border
		for (unsigned int localNodeId = 0u; localNodeId < subgraphSize; ++localNodeId)
		{
			unsigned int nodeId = outNodeIds[subgraphStartLocation + localNodeId];
			bool allNeighborsIncluded = true;
			for (unsigned int localNeighborId = adjIntervals[nodeId]; localNeighborId < adjIntervals[nodeId + 1u]; ++localNeighborId)
			{
				unsigned int neighborId = neighborIds[localNeighborId];
				bool alreadyIncluded = false;
				for (unsigned int previousNodeId = 0u; previousNodeId < subgraphSize; ++previousNodeId)
				{
					if (outNodeIds[subgraphStartLocation + previousNodeId] == neighborId)
					{
						alreadyIncluded = true;
						break;
					}
				}
				if (!alreadyIncluded)
				{
					allNeighborsIncluded = false;
					break;
				}
			}

			if (allNeighborsIncluded)
			{
				outBorderNodeFlags[subgraphStartLocation + localNodeId] = 0u;//mark as interior
			}
			else
			{
				outBorderNodeFlags[subgraphStartLocation + localNodeId] = 1u;//mark as border node
			}
		}

		//compute inner border
		for (unsigned int localNodeId = 0u; localNodeId < subgraphSize; ++localNodeId)
		{
			unsigned int nodeId = outNodeIds[subgraphStartLocation + localNodeId];
			bool nextToBorder = false;
			for (unsigned int localNeighborId = adjIntervals[nodeId]; localNeighborId < adjIntervals[nodeId + 1u] && !nextToBorder; ++localNeighborId)
			{
				unsigned int neighborId = neighborIds[localNeighborId];
				for (unsigned int previousNodeId = 0u; previousNodeId < subgraphSize; ++previousNodeId)
				{
					if (outNodeIds[subgraphStartLocation + previousNodeId] == neighborId &&
						outBorderNodeFlags[subgraphStartLocation + previousNodeId] == 1u)
					{
						outBorderNodeFlags[subgraphStartLocation + localNodeId] = 2u;//mark as inner border node
						nextToBorder = true;
						break;
					}
				}
			}
		}//end for each node in the subgraph

	}//end operator()

};

class CutMatching
{
public:
	unsigned int graphSize1;
	unsigned int graphSize2;
	unsigned int subgraphSize;
	unsigned int numSubgraphs;
	unsigned int subgraphsPerSeedNode;
	float spatialTolerance;
	//graph 1
	thrust::device_ptr<unsigned int> inNodeTypes;
	//subrgaphs 1
	thrust::device_ptr<unsigned int> inNodeIds;
	thrust::device_ptr<unsigned int> inBorderNodeFlags;
	//node-node distances 1
	thrust::device_ptr<float>        inDistMatrix;
	//node sizes 1
	thrust::device_ptr<float>        inNodeSizes;
	//graph 2
	thrust::device_ptr<unsigned int> outNodeTypes;
	//subrgaphs 2
	thrust::device_ptr<unsigned int> outNodeIds;
	thrust::device_ptr<unsigned int> outBorderNodeFlags;
	//node-node distances 2
	thrust::device_ptr<float>        outDistMatrix;
	//node types 2
	thrust::device_ptr<float>        outNodeSizes;

	thrust::device_ptr<unsigned int> outValidSubgraphFlags;

	CutMatching(
		unsigned int aGraphSize1,
		unsigned int aGraphSize2,
		unsigned int aSampleSize,
		unsigned int aNumSamples,
		float aSpatialTolerance,
		thrust::device_ptr<unsigned int> inTypes,
		thrust::device_ptr<unsigned int> inIds,
		thrust::device_ptr<unsigned int> inFlags,
		thrust::device_ptr<float> inMatrix,
		thrust::device_ptr<float> inSizes,
		thrust::device_ptr<unsigned int> outTypes,
		thrust::device_ptr<unsigned int> outIds,
		thrust::device_ptr<unsigned int> outFlags,
		thrust::device_ptr<float> outMatrix,
		thrust::device_ptr<float> outSizes,
		thrust::device_ptr<unsigned int> outSubgraphFlags
	) : graphSize1(aGraphSize1),
		graphSize2(aGraphSize2),
		subgraphSize(aSampleSize),
		numSubgraphs(aNumSamples),
		subgraphsPerSeedNode(aNumSamples / aGraphSize1),
		spatialTolerance(aSpatialTolerance),
		inNodeTypes(inTypes),
		inNodeIds(inIds),
		inBorderNodeFlags(inFlags),
		inDistMatrix(inMatrix),
		inNodeSizes(inSizes),
		outNodeTypes(outTypes),
		outNodeIds(outIds),
		outBorderNodeFlags(outFlags),
		outDistMatrix(outMatrix),
		outNodeSizes(outSizes),
		outValidSubgraphFlags(outSubgraphFlags)
	{}
	
	//__host__ __device__ FORCE_INLINE void invalidateSubgraph(unsigned int subgraphStartLocation)
	//{
	//	for (unsigned int localNodeId = 0u; localNodeId < subgraphSize; ++localNodeId)
	//	{
	//		outNodeIds[subgraphStartLocation + localNodeId] = graphSize2;
	//		outBorderNodeFlags[subgraphStartLocation + localNodeId] = 0u;
	//	}
	//}

	__host__ __device__	bool randomMatchingOperator(unsigned int aId, unsigned int aRandOffset)
	{
		unsigned int subgraphStartLocation = aId * subgraphSize;// (aId % 32) * subgraphSize;

		unsigned int interiorNodesCount = 0;
		for (unsigned int localNodeId = 0u; localNodeId < subgraphSize; ++localNodeId)
		{
			if (inBorderNodeFlags[subgraphStartLocation + localNodeId] == 0u)
			{
				//only match the nodes on the inner border and border
				outNodeIds[subgraphStartLocation + localNodeId] = graphSize2;
				++interiorNodesCount;
			}
			else
			{
				outNodeIds[subgraphStartLocation + localNodeId] = aRandOffset;
			}
			outBorderNodeFlags[subgraphStartLocation + localNodeId] = 0u;
		}

		if (interiorNodesCount == 0 || subgraphSize - interiorNodesCount < 3)
		{
			//invalid subgraph - too few nodes in the cut, or no interior nodes
			//invalidateSubgraph(subgraphStartLocation);
			return false;
		}


		for (unsigned int localNodeId = 0u; localNodeId < subgraphSize; ++localNodeId)
		{
			if (inBorderNodeFlags[subgraphStartLocation + localNodeId] == 0u)
				continue;

			unsigned int inNodeId = inNodeIds[subgraphStartLocation + localNodeId];
			unsigned int inNodeType = inNodeTypes[inNodeId];
			bool foundAMatch = false;
			for (unsigned int nodeId2 = outNodeIds[subgraphStartLocation + localNodeId]; nodeId2 < graphSize2 && !foundAMatch; ++nodeId2)
			{
				//unsigned int nodeId2RND = (randNodeIdOffset + nodeId2) % graphSize2;
				//unsigned int nodeId2 = (nodeId2_it + 16) % graphSize2;
				bool matches = outNodeTypes[nodeId2] == inNodeType;
				if (!matches)
					continue;
				bool skip = false;
				for (unsigned int recordedNodeId = 0; recordedNodeId < localNodeId && !skip; ++recordedNodeId)
				{
					if (outNodeIds[subgraphStartLocation + recordedNodeId] == nodeId2)
						skip = true;//already participates
					if (inBorderNodeFlags[subgraphStartLocation + recordedNodeId] == 0u)
						continue;//skip interior nodes
					unsigned int pairNodeId = inNodeIds[subgraphStartLocation + recordedNodeId];
					const float targetDist = inDistMatrix[inNodeId + graphSize1 * pairNodeId];
					unsigned int pairNodeId2 = outNodeIds[subgraphStartLocation + recordedNodeId];
					const float currentDist = outDistMatrix[nodeId2 + graphSize2 * pairNodeId2];
					if (fabsf(currentDist - targetDist) > spatialTolerance /** outNodeSizes[pairNodeId2]*/)
						skip = true;//incompatible with previous participants
				}
				if (skip)
					continue;

				for (unsigned int pairingNodeId = 0; pairingNodeId < subgraphSize; ++pairingNodeId)
				{
					if (inBorderNodeFlags[subgraphStartLocation + pairingNodeId] == 0u)
						continue;
					if (pairingNodeId == localNodeId)
						continue;
					unsigned int pairNodeId = inNodeIds[subgraphStartLocation + pairingNodeId];
					unsigned int pairNodeType = inNodeTypes[pairNodeId];
					const float targetDist = inDistMatrix[inNodeId + graphSize1 * pairNodeId];
					bool foundMatchingPair = false;
					for (unsigned int pairingNodeId2 = 0u; pairingNodeId2 < graphSize2 && !foundMatchingPair; ++pairingNodeId2)
					{
						if (pairingNodeId2 == nodeId2)
							continue;
						if (outNodeTypes[pairingNodeId2] != pairNodeType)
							continue;
						const float currentDist = outDistMatrix[nodeId2 + graphSize2 * pairingNodeId2];
						if (fabsf(currentDist - targetDist) < spatialTolerance /** inNodeSizes[pairingNodeId2]*/)
							foundMatchingPair = true;
					}//end for all other nodes in the second graph
					if (!foundMatchingPair)
						matches = false;
				}//end for all other nodes in the cut
				if (matches)
				{
					outNodeIds[subgraphStartLocation + localNodeId] = nodeId2;
					outBorderNodeFlags[subgraphStartLocation + localNodeId] = inBorderNodeFlags[subgraphStartLocation + localNodeId];
					foundAMatch = true;
				}
			}//end for all nodes in the second graph
			if (!foundAMatch)
			{
				//backtrack
				if (localNodeId > 0) --localNodeId;
				while (inBorderNodeFlags[subgraphStartLocation + localNodeId] == 0u && localNodeId > 0) --localNodeId;

				if (localNodeId <= 1)
				{
					//did not find a mathcing node in the second graph, invalidate the subgraph
					//invalidateSubgraph(subgraphStartLocation);
					return false;
				}

				outNodeIds[subgraphStartLocation + localNodeId] = outNodeIds[subgraphStartLocation + localNodeId] + 1;
				outBorderNodeFlags[subgraphStartLocation + localNodeId] = 0u;
				--localNodeId;//cancel out post-increment from the loop
			}
		}//end for all nodes in the cut

     	//double check selected matching nodes
		bool foundMismatch = false;
		for (unsigned int localNodeId = 0u; localNodeId < subgraphSize && !foundMismatch; ++localNodeId)
		{
			if (inBorderNodeFlags[subgraphStartLocation + localNodeId] == 0u)
				continue;

			unsigned int inNodeId = inNodeIds[subgraphStartLocation + localNodeId];
			unsigned int inNodeType = inNodeTypes[inNodeId];
			unsigned int outNodeId = outNodeIds[subgraphStartLocation + localNodeId];
			unsigned int outNodeType = outNodeTypes[outNodeId];
			if (inNodeType != outNodeType)
			{
				foundMismatch = true;
				break;
			}

			for (unsigned int pairingNodeId = 0; pairingNodeId < subgraphSize; ++pairingNodeId)
			{
				if (inBorderNodeFlags[subgraphStartLocation + pairingNodeId] == 0u)
					continue;
				if (pairingNodeId == localNodeId)
					continue;
				unsigned int inPairNodeId = inNodeIds[subgraphStartLocation + pairingNodeId];
				unsigned int outPairNodeId = outNodeIds[subgraphStartLocation + pairingNodeId];
				const float targetDist = inDistMatrix[inNodeId + graphSize1 * inPairNodeId];
				const float currentDist = outDistMatrix[outNodeId + graphSize2 * outPairNodeId];
				if (fabsf(currentDist - targetDist) > spatialTolerance * outNodeSizes[outPairNodeId])
				{
					foundMismatch = true;
					break;
				}
			}//end for each other node in the cut
		}//end for each node in the cut

		if (!foundMismatch)
		{
			outValidSubgraphFlags[aId] = 1u;
			return true;
		}

		return false;
	}


	__host__ __device__	void operator()(const size_t& aId_s)
	{
		unsigned int aId = (unsigned int)aId_s;
		//unsigned int subgraphSeedNodeId = subgraphsPerSeedNode == 0u ? aId : aId / subgraphsPerSeedNode;
		//unsigned int subgraphOffset = subgraphsPerSeedNode == 0u ? 0u : aId % subgraphsPerSeedNode;
		//unsigned int subgraphStartLocation = subgraphOffset * subgraphSize + subgraphSeedNodeId * subgraphsPerSeedNode * subgraphSize;
		
		KISSRandomNumberGenerator genRand(
			3643u + aId * 4154207u * graphSize2 + graphSize2,
			1761919u + aId * 2746753u * graphSize1,
			331801u + aId,
			10499029u);

		unsigned int offset = (unsigned int)(genRand() * (float)graphSize2);
		bool success = randomMatchingOperator(aId, offset);
		//if(!success)
		//	success = randomMatchingOperator(aId, offset / 2u);
		//if(!success)
		//	randomMatchingOperator(aId, offset / 4u);
		if(!success)
			randomMatchingOperator(aId, 0u);

	}


};

class TransformationEstimator
{
public:
	unsigned int subgraphSize;

	thrust::device_ptr<float3> positions1;
	thrust::device_ptr<float3> positions2;

	thrust::device_ptr<unsigned int> nodeIds1;
	thrust::device_ptr<unsigned int> borderNodeFlags;
	thrust::device_ptr<unsigned int> nodeIds2;

	thrust::device_ptr<unsigned int> outValidSubgraphFlags;

	thrust::device_ptr<float3> outTranslation1;
	thrust::device_ptr<float3> outTranslation2;
	thrust::device_ptr<float> tmpCovMatrix;
	thrust::device_ptr<float> tmpDiagonalW;
	thrust::device_ptr<float> tmpMatrixV;
	thrust::device_ptr<float> tmpVecRV;
	thrust::device_ptr<quaternion4f> outRotation2;


	TransformationEstimator(
		unsigned int aSampleSize,
		thrust::device_ptr<float3> aPositions1,
		thrust::device_ptr<float3> aPositions2,
		thrust::device_ptr<unsigned int> inIds,
		thrust::device_ptr<unsigned int> inFlags,
		thrust::device_ptr<unsigned int> outIds,
		thrust::device_ptr<unsigned int> outSubgraphFlags,
		thrust::device_ptr<float3> aTranslation1,
		thrust::device_ptr<float3> aTranslation2,
		thrust::device_ptr<float> aCovMatrix,
		thrust::device_ptr<float> aDiagonalW,
		thrust::device_ptr<float> aMatrixV,
		thrust::device_ptr<float> aVecRV,
		thrust::device_ptr<quaternion4f> aOutRot
	) : subgraphSize(aSampleSize),
		positions1(aPositions1),
		positions2(aPositions2),
		nodeIds1(inIds),
		borderNodeFlags(inFlags),
		nodeIds2(outIds),
		outValidSubgraphFlags(outSubgraphFlags),
		outTranslation1(aTranslation1),
		outTranslation2(aTranslation2),
		tmpCovMatrix(aCovMatrix),
		tmpDiagonalW(aDiagonalW),
		tmpMatrixV(aMatrixV),
		tmpVecRV(aVecRV),
		outRotation2(aOutRot)
	{}

	__host__ __device__	void operator()(const size_t& aId_s)
	{
		unsigned int aId = (unsigned int)aId_s;
		if (outValidSubgraphFlags[aId] == 0u)
			return;

		unsigned int subgraphStartLocation = aId * subgraphSize;// (aId % 32) * subgraphSize;

		//Compute the means of the border node locations
		float3 center1 = make_float3(0.f, 0.f, 0.f);
		float3 center2 = make_float3(0.f, 0.f, 0.f);
		float numPoints = 0.f;
		for (unsigned int i = 0u; i < subgraphSize; ++i)
		{
			if (borderNodeFlags[subgraphStartLocation + i] != 0u)
			{
				center1 += positions1[nodeIds1[subgraphStartLocation + i]];
				center2 += positions2[nodeIds2[subgraphStartLocation + i]];
				numPoints += 1.f;
			}
		}
		center1 /= numPoints;
		center2 /= numPoints;

		//Compute covariance matrix
		float* covMat = thrust::raw_pointer_cast(tmpCovMatrix + aId * 3 * 3);
		for (unsigned int i = 0u; i < subgraphSize; ++i)
		{
			if (borderNodeFlags[subgraphStartLocation + i] != 0u)
			{
				float3 vec1 = positions1[nodeIds1[subgraphStartLocation + i]] - center1;
				float3 vec2 = positions2[nodeIds2[subgraphStartLocation + i]] - center2;

				covMat[0 * 3 + 0] += vec2.x * vec1.x;
				covMat[1 * 3 + 0] += vec2.y * vec1.x;
				covMat[2 * 3 + 0] += vec2.z * vec1.x;

				covMat[0 * 3 + 1] += vec2.x * vec1.y;
				covMat[1 * 3 + 1] += vec2.y * vec1.y;
				covMat[2 * 3 + 1] += vec2.z * vec1.y;

				covMat[0 * 3 + 2] += vec2.x * vec1.z;
				covMat[1 * 3 + 2] += vec2.y * vec1.z;
				covMat[2 * 3 + 2] += vec2.z * vec1.z;
			}
		}
		//Singular Value Decomposition
		float* diag = thrust::raw_pointer_cast(tmpDiagonalW + aId * 3);
		float* vMat = thrust::raw_pointer_cast(tmpMatrixV + aId * 3 * 3);
		float* tmp = thrust::raw_pointer_cast(tmpVecRV + aId * 3);

		svd::svdcmp(covMat, 3, 3, diag, vMat, tmp);

		//Rotation is V * transpose(U)		
		for (unsigned int row = 0; row < 3; ++row)
		{
			for (unsigned int col = 0; col < 3; ++col)
			{
				tmp[col] =
					vMat[row * 3 + 0] * covMat[col * 3 + 0] +
					vMat[row * 3 + 1] * covMat[col * 3 + 1] +
					vMat[row * 3 + 2] * covMat[col * 3 + 2];
			}
			vMat[row * 3 + 0] = tmp[0];
			vMat[row * 3 + 1] = tmp[1];
			vMat[row * 3 + 2] = tmp[2];
		}


		float rotDet = determinant(
			vMat[0], vMat[3], vMat[6],
			vMat[1], vMat[4], vMat[7],
			vMat[2], vMat[5], vMat[8]
		);

		if (rotDet < 0.f)
		{
			vMat[6] = -vMat[6];
			vMat[7] = -vMat[7];
			vMat[8] = -vMat[8];
			rotDet = -rotDet;
		}

		if (fabsf(rotDet - 1.f)> EPS)
			outValidSubgraphFlags[aId] = 0u;


		quaternion4f rotation(
			vMat[0], vMat[3], vMat[6],
			vMat[1], vMat[4], vMat[7],
			vMat[2], vMat[5], vMat[8]
		);
		outTranslation1[aId] = center1;
		outTranslation2[aId] = center2;
		outRotation2[aId] = rotation;
	}

};


__host__ std::string VariationGenerator::operator()(const char * aFilePath1, const char * aFilePath2,
	WFObject & aObj1, WFObject & aObj2, Graph & aGraph1, Graph & aGraph2, float aRelativeThreshold)
{
	cudastd::timer timer;
	cudastd::timer intermTimer;

	if (aGraph1.numNodes() < 9u || aGraph2.numNodes() < 9u)
		return "";

	thrust::host_vector<float3> objCenters1;
	thrust::host_vector<float> objSizes1;

	ObjectCenterExporter()(aObj1, objCenters1, objSizes1, 0.3333f);

	thrust::host_vector<float3> objCenters2;
	thrust::host_vector<float> objSizes2;

	ObjectCenterExporter()(aObj2, objCenters2, objSizes2, 0.3333f);

	thrust::device_vector<float3> centersDevice1(objCenters1);
	thrust::device_vector<float> pairwiseDistMatrix1(objCenters1.size() * objCenters1.size());
	thrust::counting_iterator<size_t> first(0u);
	thrust::counting_iterator<size_t> last1(pairwiseDistMatrix1.size());
	DistanceMatrixWriter writeDistances1(objCenters1.size(), pairwiseDistMatrix1.data(), centersDevice1.data());
	thrust::for_each(first, last1, writeDistances1);

	thrust::host_vector<unsigned int> nodeTypes1Host(aGraph1.numNodes(), (unsigned int)aObj1.materials.size());
	for (size_t nodeId = 0; nodeId < aObj1.objects.size(); ++nodeId)
	{
		size_t faceId = aObj1.objects[nodeId].x;
		size_t materialId = aObj1.faces[faceId].material;
		nodeTypes1Host[nodeId] = (unsigned int)materialId;
	}
	thrust::device_vector<unsigned int> nodeTypes1(nodeTypes1Host);

	thrust::device_vector<float3> centersDevice2(objCenters2);
	thrust::device_vector<float> pairwiseDistMatrix2(objCenters2.size() * objCenters2.size());
	thrust::counting_iterator<size_t> last2(pairwiseDistMatrix2.size());
	DistanceMatrixWriter writeDistances2(objCenters2.size(), pairwiseDistMatrix2.data(), centersDevice2.data());
	thrust::for_each(first, last2, writeDistances2);

	thrust::host_vector<unsigned int> nodeTypes2Host(aGraph2.numNodes(), (unsigned int)aObj2.materials.size());
	for (size_t nodeId = 0u; nodeId < aObj2.objects.size(); ++nodeId)
	{
		size_t faceId = aObj2.objects[nodeId].x;
		size_t materialId = aObj2.faces[faceId].material;
		nodeTypes2Host[nodeId] = (unsigned int)materialId;
	}
	thrust::device_vector<unsigned int> nodeTypes2(nodeTypes2Host);

	float3 minBound, maxBound;
	ObjectBoundsExporter()(aObj1, minBound, maxBound);
	const float boundsDiagonal = len(maxBound - minBound);
	const float spatialTolerance = boundsDiagonal * std::max(aRelativeThreshold, 0.02f);
	//const float spatialTolerance = 30.f * (aRelativeThreshold + 0.03f);


	std::string result = "";

	initTime = intermTimer.get();
	intermTimer.start();

	samplingTime = matchingTime = svdTime = matchingTime = 0.f;
	histTime = transformTime = collisionTime = exportTime = conversionTime = 0.f;
	histoChecks = matchingCuts = matchingCutsAndTs = histoChecksPassed = 0u;

	std::vector<NodeTypeHistogram> variatioHistograms;
	variatioHistograms.push_back(NodeTypeHistogram(nodeTypes1Host));
	variatioHistograms.push_back(NodeTypeHistogram(nodeTypes2Host));

	thrust::host_vector<unsigned int> graph2Intervals(aGraph2.intervals);
	thrust::host_vector<unsigned int> graph2NbrIds(aGraph2.adjacencyVals);
	thrust::host_vector<unsigned int> graph1Intervals(aGraph1.intervals);
	thrust::host_vector<unsigned int> graph1NbrIds(aGraph1.adjacencyVals);

	thrust::device_vector<float> objSizes1Device(objSizes1);
	thrust::device_vector<float> objSizes2Device(objSizes2);

	GrammarCheck grammarCheck;
	grammarCheck.init(graph2Intervals, graph2NbrIds, nodeTypes2Host);
	grammarCheck.init(graph1Intervals, graph1NbrIds, nodeTypes1Host);

	numVariations = 0u;

	const unsigned int numSubgraphSamples = 32u * (unsigned int)aGraph1.numNodes();// std::max(aGraph1.numNodes(), aGraph2.numNodes());
	//const unsigned int subgraphSampleSize = (unsigned int)objCenters1.size() / 2u;

	for (unsigned int subgraphSampleSize = 4u;// (unsigned int)std::min(aGraph1.numNodes(), aGraph2.numNodes()) / 4u;
		subgraphSampleSize < (unsigned int) 3u * aGraph1.numNodes() / 4u;
		subgraphSampleSize++)
	{
		std::cout << "Subgraph sample size: " << subgraphSampleSize << " / " << 3u * aGraph1.numNodes() / 4u <<"\r";

		if (subgraphSampleSize < 3)
			continue;

		thrust::device_vector<unsigned int> subgraphNodeIds1(numSubgraphSamples * subgraphSampleSize);
		thrust::device_vector<unsigned int> subgraphBorderFlags1(numSubgraphSamples * subgraphSampleSize);

		thrust::device_vector<unsigned int> subgraphNodeIds2(numSubgraphSamples * subgraphSampleSize);
		thrust::device_vector<unsigned int> subgraphBorderFlags2(numSubgraphSamples * subgraphSampleSize);

		SubgraphInitializer initSubgraphSamples(
			(unsigned int)objCenters1.size(),
			subgraphSampleSize,
			numSubgraphSamples,
			aGraph1.intervals.data(),
			aGraph1.adjacencyVals.data(),
			subgraphNodeIds1.data(),
			subgraphBorderFlags1.data());

		thrust::counting_iterator<size_t> lastSubgraph(numSubgraphSamples);
		thrust::for_each(first, lastSubgraph, initSubgraphSamples);

		samplingTime += intermTimer.get();
		intermTimer.start();

		//#ifdef _DEBUG
		//	outputDeviceVector("Subgraph node ids     1: ", subgraphNodeIds1);
		//	outputDeviceVector("Subgraph border flags 1: ", subgraphBorderFlags1);
		//#endif

		///////////////////////////////////////////////////////////////////////////////////
		//Find matching cuts in both sub-graphs

		thrust::device_vector<unsigned int> validSubgraphFlags(numSubgraphSamples, 0u);

		CutMatching matchCuts(
			(unsigned int)objCenters1.size(),
			(unsigned int)objCenters2.size(),
			subgraphSampleSize,
			numSubgraphSamples,
			spatialTolerance,
			nodeTypes1.data(),
			subgraphNodeIds1.data(),
			subgraphBorderFlags1.data(),
			pairwiseDistMatrix1.data(),
			objSizes1Device.data(),
			nodeTypes2.data(),
			subgraphNodeIds2.data(),
			subgraphBorderFlags2.data(),
			pairwiseDistMatrix2.data(),
			objSizes2Device.data(),
			validSubgraphFlags.data()
		);

		//thrust::counting_iterator<size_t> lastSubgraphDbg(4);
		thrust::for_each(first, lastSubgraph, matchCuts);

		matchingCuts += thrust::count(validSubgraphFlags.begin(), validSubgraphFlags.end(), 1u);

		matchingTime += intermTimer.get();
		intermTimer.start();

		//#ifdef _DEBUG
		//	outputDeviceVector("Subgraph node ids     2: ", subgraphNodeIds2);
		//	outputDeviceVector("Subgraph border flags 2: ", subgraphBorderFlags2);
		//	outputDeviceVector("Valid subgraph flags   : ", validSubgraphFlags);
		//#endif


		///////////////////////////////////////////////////////////////////////////////////
		//Find correspondence transformation between both sub-graphs
		thrust::device_vector<float3> outTranslation1(numSubgraphSamples);
		thrust::device_vector<float3> outTranslation2(numSubgraphSamples);
		thrust::device_vector<float> tmpCovMatrix(numSubgraphSamples * 3 * 3, 0.f);
		thrust::device_vector<float> tmpDiagonalW(numSubgraphSamples * 3);
		thrust::device_vector<float> tmpMatrixV(numSubgraphSamples * 3 * 3);
		thrust::device_vector<float> tmpVecRV(numSubgraphSamples * 3);
		thrust::device_vector<quaternion4f> outRotation2(numSubgraphSamples);

		TransformationEstimator estimateT(
			subgraphSampleSize,
			centersDevice1.data(),
			centersDevice2.data(),
			subgraphNodeIds1.data(),
			subgraphBorderFlags1.data(),
			subgraphNodeIds2.data(),
			validSubgraphFlags.data(),
			outTranslation1.data(),
			outTranslation2.data(),
			tmpCovMatrix.data(),
			tmpDiagonalW.data(),
			tmpMatrixV.data(),
			tmpVecRV.data(),
			outRotation2.data()
		);

		thrust::for_each(first, lastSubgraph, estimateT);

		matchingCutsAndTs += thrust::count(validSubgraphFlags.begin(), validSubgraphFlags.end(), 1u);

		svdTime += intermTimer.get();
		intermTimer.start();

		///////////////////////////////////////////////////////////////////////////////////
		//Copy back to host
		thrust::host_vector<unsigned int> subgraphNodeIdsHost1(subgraphNodeIds1);
		thrust::host_vector<unsigned int> subgraphBorderFlagsHost1(subgraphBorderFlags1);

		thrust::host_vector<unsigned int> subgraphNodeIdsHost2(subgraphNodeIds2);
		thrust::host_vector<unsigned int> subgraphBorderFlagsHost2(subgraphBorderFlags2);

		thrust::host_vector<unsigned int> validSubgraphFlagsHost(validSubgraphFlags);

		thrust::host_vector<float3> outTranslation1Host(outTranslation1);
		thrust::host_vector<float3> outTranslation2Host(outTranslation2);
		thrust::host_vector<quaternion4f> outRotation2Host(outRotation2);

		unsigned int graphSize1 = (unsigned int)objCenters1.size();
		unsigned int graphSize2 = (unsigned int)objCenters2.size();

		GraphToStringConverter convertToStr;
		CollisionGraphExporter graphExporter;
		WFObjectFileExporter   objExporter;

		cpyBackTime += intermTimer.get();
		intermTimer.start();

		for (unsigned int subgraphId = 0u; subgraphId < numSubgraphSamples; ++subgraphId)
		{
			intermTimer.start();

			if (validSubgraphFlagsHost[subgraphId] != 1u)
				continue;

			thrust::host_vector<unsigned int> completeSubgraphFlags2(graphSize2, 0u);
			std::vector<unsigned int> nodeStack;
			unsigned int subgraph2Size = 0u;
			unsigned int complementSize = 0u;
			thrust::host_vector<unsigned int>::iterator subgraphNodeIdsHost1Begin = subgraphNodeIdsHost1.begin() + subgraphId * subgraphSampleSize;
			thrust::host_vector<unsigned int>::iterator subgraphNodeIdsHost2Begin = subgraphNodeIdsHost2.begin() + subgraphId * subgraphSampleSize;
			thrust::host_vector<unsigned int>::iterator subgraphBorderFlagsHost1Begin = subgraphBorderFlagsHost1.begin() + subgraphId * subgraphSampleSize;
			thrust::host_vector<unsigned int>::iterator subgraphBorderFlagsHost2Begin = subgraphBorderFlagsHost2.begin() + subgraphId * subgraphSampleSize;

			//initialize flags at graph cut - 2 -> outside node, 1 -> border node
			for (unsigned int i = 0u; i < subgraphSampleSize; ++i)
			{
				if (*(subgraphBorderFlagsHost2Begin + i) != 0u)
					completeSubgraphFlags2[*(subgraphNodeIdsHost2Begin + i)] = *(subgraphBorderFlagsHost2Begin + i);
				if (*(subgraphBorderFlagsHost2Begin + i) == 1u)
				{
					++subgraph2Size;
					nodeStack.push_back(*(subgraphNodeIdsHost2Begin + i));
				}
				if (*(subgraphBorderFlagsHost2Begin + i) == 2u)
					++complementSize;
			}
			//region grow from each border node
			while (!nodeStack.empty())
			{
				unsigned int nodeId = nodeStack.back();
				nodeStack.pop_back();
				for (unsigned int nbr = graph2Intervals[nodeId]; nbr < graph2Intervals[nodeId + 1]; ++nbr)
				{
					unsigned int nbrId = graph2NbrIds[nbr];
					if (completeSubgraphFlags2[nbrId] == 2u || completeSubgraphFlags2[nbrId] == 1u)
						continue;
					completeSubgraphFlags2[nbrId] = 1u;
					nodeStack.push_back(nbrId);
					++subgraph2Size;
				}
			}
			//check validity
			if (subgraph2Size + complementSize >= graphSize2)
				continue; //should not happen
			unsigned int subgraph1Size = 0u;
			thrust::host_vector<unsigned int> completeSubgraphFlags1(graphSize1, 0u);
			for (unsigned int i = 0u; i < subgraphSampleSize; ++i)
			{
				if (*(subgraphBorderFlagsHost1Begin + i) == 0u || *(subgraphBorderFlagsHost1Begin + i) == 2u)
				{
					completeSubgraphFlags1[*(subgraphNodeIdsHost1Begin + i)] = 1u;
					++subgraph1Size;
				}
			}

			///////////////////////////////////////////////////////////////////////////////////
			//discard variations with repeating node type histograms
			variatioHistograms.push_back(NodeTypeHistogram(aObj1.materials.size()));
			for (auto inTypeIt1 = nodeTypes1Host.begin(); inTypeIt1 != nodeTypes1Host.end(); ++inTypeIt1)
			{
				if (completeSubgraphFlags1[inTypeIt1 - nodeTypes1Host.begin()] == 1u)
				{
					variatioHistograms.back().typeCounts[*inTypeIt1]++;
				}
			}
			for (auto inTypeIt2 = nodeTypes2Host.begin(); inTypeIt2 != nodeTypes2Host.end(); ++inTypeIt2)
			{
				if (completeSubgraphFlags2[inTypeIt2 - nodeTypes2Host.begin()] == 1u)
				{
					variatioHistograms.back().typeCounts[*inTypeIt2]++;
				}
			}

			bool repeatedHistogram = false;
			for (size_t hid = 0u; hid < variatioHistograms.size() - 1 && !repeatedHistogram; ++hid)
			{
				++histoChecks;
				if (variatioHistograms.back() == variatioHistograms[hid])
					repeatedHistogram = true;
			}


			histTime += intermTimer.get();
			intermTimer.start();

			if (repeatedHistogram)
			{
				variatioHistograms.pop_back();
				continue;
			}

			++histoChecksPassed;
			////////////////////////////////////////////////////////////////////////////////////////

			for (unsigned int i = 0u; i < graphSize2; ++i)
			{
				if (completeSubgraphFlags2[i] == 2u)
					completeSubgraphFlags2[i] = 0u;
			}

			//graphExporter.exportSubGraph(aFilePath1, aObj1, aGraph1, numVariations, completeSubgraphFlags1);
			//graphExporter.exportSubGraph(aFilePath2, aObj2, aGraph2, numVariations, completeSubgraphFlags2);

			///////////////////////////////////////////////////////////////////////////////////
			//Create the variation by merging the subsets of aObj1 and aObj2
			float3 translation1 = outTranslation1Host[subgraphId];
			float3 translation2 = outTranslation2Host[subgraphId];
			quaternion4f rotation2 = outRotation2Host[subgraphId];
			WFObject variation = WFObjectMerger()(aObj1, translation1, aObj2, translation2, rotation2, completeSubgraphFlags1, completeSubgraphFlags2);
			///////////////////////////////////////////////////////////////////////////////////
			transformTime += intermTimer.get();
			intermTimer.start();
			///////////////////////////////////////////////////////////////////////////////////
			//Compute the collision graph for the variation
			CollisionDetector detector;
			Graph variationGraph = detector.computeCollisionGraph(variation, aRelativeThreshold);
			///////////////////////////////////////////////////////////////////////////////////
			collisionTime += intermTimer.get();
			intermTimer.start();
			///////////////////////////////////////////////////////////////////////////////////
			//Check that the variation graph is valid
			thrust::host_vector<unsigned int> nodeTypesVariation(variationGraph.numNodes());
			for (size_t nodeId = 0; nodeId < variationGraph.numNodes(); ++nodeId)
			{
				size_t faceId = variation.objects[nodeId].x;
				size_t materialId = variation.faces[faceId].material;
				nodeTypesVariation[nodeId] = (unsigned int)materialId;
			}
			thrust::host_vector<unsigned int> hostIntervals(variationGraph.intervals);
			thrust::host_vector<unsigned int> hostNbrIds(variationGraph.adjacencyVals);
			if (!grammarCheck.check(hostIntervals, hostNbrIds, nodeTypesVariation))
			{
				//variationGraph = detector.computeCollisionGraph(variation, std::max(aRelativeThreshold, 0.02f));
				//hostIntervals = variationGraph.intervals;
				//hostNbrIds = variationGraph.adjacencyVals;
				//if (!grammarCheck.check(hostIntervals, hostNbrIds, nodeTypesVariation))
				//{
					variatioHistograms.pop_back();
					continue;
				//}

			}
			///////////////////////////////////////////////////////////////////////////////////
			++numVariations;

			if (writeVariationGraphs || writeVariations)
			{
				std::string fileName1(aFilePath1);
				if (fileName1.find_last_of("/\\") == std::string::npos)
					fileName1 = fileName1.substr(0, fileName1.size() - 5);
				else
					fileName1 = fileName1.substr(fileName1.find_last_of("/\\") + 1, fileName1.size() - fileName1.find_last_of("/\\") - 5);

				std::string fileName2(aFilePath2);
				if (fileName2.find_last_of("/\\") == std::string::npos)
					fileName2 = fileName2.substr(0, fileName2.size() - 5);
				else
					fileName2 = fileName2.substr(fileName2.find_last_of("/\\") + 1, fileName2.size() - fileName2.find_last_of("/\\") - 5);

				std::string objDir = getDirName(aFilePath2);
				std::string variationFilePath = objDir + fileName1 + "_" + fileName2 + "_" + itoa((int)numVariations);
				if (writeVariations)
					objExporter(variation, variationFilePath.c_str());
				
				if (writeVariationGraphs)
					graphExporter.exportCollisionGraph((variationFilePath + ".obj").c_str(), variation, variationGraph);
			}

			exportTime = intermTimer.get();
			intermTimer.start();

			std::string variationStrings = convertToStr(variation, variationGraph);
			if (!multiString)
				variationStrings = variationStrings.substr(0u, variationStrings.find_first_of("\n"));
			result.append(variationStrings);

			conversionTime += intermTimer.get();
		}//end for subgraph samples

	}//end for subgraph size

	std::cout << "Total subgraph samples: " << numSubgraphSamples * (unsigned int)3u * aGraph1.numNodes() / 4 << "\n";

	totalTime = timer.get();

	intermTimer.cleanup();
	timer.cleanup();

	return result;
}


__host__ void VariationGenerator::stats()
{
	size_t miliseconds = (size_t)totalTime % 1000u;
	size_t minutes = (size_t)totalTime / 60000;
	size_t seconds = ((size_t)totalTime % 60000) / 1000;

	std::cerr << "Created "<< numVariations <<" variations in " << totalTime << "ms "
		<< minutes << ":" << seconds << ":" << miliseconds << " (min:sec:ms)\n";
	std::cerr << "Matching subgraph cuts   : " << matchingCuts << "\n";
	std::cerr << "Matching transformations : " << matchingCutsAndTs << "\n";
	std::cerr << "New histograms           : " << histoChecksPassed << "\n";
	std::cerr << "Grammar checks passed    : " << numVariations << "\n";
	std::cerr << "-------------------------------------\n";
	std::cerr << "Initialization in      " << initTime << "ms\n";
	std::cerr << "Subgraph sampling in   " << samplingTime << "ms\n";
	std::cerr << "Graph cut matching in  " << matchingTime << "ms\n";
	std::cerr << "SVD in                 " << svdTime << "ms\n";
	std::cerr << "Mem transfer in        " << cpyBackTime << "ms\n";
	std::cerr << "Histogram check  in    " << histTime << "ms (performed   " << histoChecks << " checks)\n";
	std::cerr << "Obj transformation in  " << transformTime << "ms\n";
	std::cerr << "Collision detection in " << collisionTime << "ms\n";
	std::cerr << "File export in         " << exportTime << "ms\n";
	std::cerr << "String conversion      " << conversionTime << "ms\n";
}