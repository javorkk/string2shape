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
		unsigned int subgraphOffset = subgraphsPerSeedNode == 0u ? aId : aId % subgraphsPerSeedNode;
		unsigned int subgraphStartLocation = subgraphOffset * subgraphSize + subgraphSeedNodeId * subgraphsPerSeedNode * subgraphSize;

		outNodeIds[subgraphStartLocation] = subgraphSeedNodeId;

		unsigned int currentSize = 1u;
		unsigned int currentDepth = 0u;
		unsigned int currentSubgraphNodeId = 0u;

		//compute subgraph
		while (currentSize < subgraphSize && currentSize < graphSize)
		{
			unsigned int neighborCount = 0u;
			for (unsigned int localNodeId = currentSubgraphNodeId; localNodeId < currentSize; ++localNodeId)
			{
				unsigned int nodeId = outNodeIds[subgraphStartLocation + localNodeId];
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

					if (!alreadyIncluded && neighborCount + currentSize < subgraphSize) //add to subgraph
					{
						outNodeIds[subgraphStartLocation + neighborCount + currentSize] = neighborId;
						neighborCount++;
					}
					else if (!alreadyIncluded && genRand() < 0.5)//replace a random node with the same depth
					{
						unsigned int randLocation = (int)(genRand() * (float)neighborCount);
						outNodeIds[subgraphStartLocation + randLocation + currentSize] = neighborId;
					}
				}

			}

			currentSubgraphNodeId = currentSize;
			currentDepth++;
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

		}
	}

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
	//thrust::device_ptr<unsigned int> inIntervals;
	//thrust::device_ptr<unsigned int> inNeighborIds;
	thrust::device_ptr<unsigned int> inNodeTypes;
	//subrgaphs 1
	thrust::device_ptr<unsigned int> inNodeIds;
	thrust::device_ptr<unsigned int> inBorderNodeFlags;
	//node-node distances 1
	thrust::device_ptr<float>        inDistMatrix;
	//graph 2
	thrust::device_ptr<unsigned int> outNodeTypes;
	//subrgaphs 2
	thrust::device_ptr<unsigned int> outNodeIds;
	thrust::device_ptr<unsigned int> outBorderNodeFlags;
	//node-node distances 2
	thrust::device_ptr<float>        outDistMatrix;

	thrust::device_ptr<unsigned int> outValidSubgraphFlags;

	CutMatching(
		unsigned int aGraphSize1,
		unsigned int aGraphSize2,
		unsigned int aSampleSize,
		unsigned int aNumSamples,
		float aSpatialTolerance,
		//thrust::device_ptr<unsigned int> aIntervals,
		//thrust::device_ptr<unsigned int> aNeighborIds,
		thrust::device_ptr<unsigned int> inTypes,
		thrust::device_ptr<unsigned int> inIds,
		thrust::device_ptr<unsigned int> inFlags,
		thrust::device_ptr<float> inMatrix,
		thrust::device_ptr<unsigned int> outTypes,
		thrust::device_ptr<unsigned int> outIds,
		thrust::device_ptr<unsigned int> outFlags,
		thrust::device_ptr<float> outMatrix,
		thrust::device_ptr<unsigned int> outSubgraphFlags
	) : graphSize1(aGraphSize1),
		graphSize2(aGraphSize2),
		subgraphSize(aSampleSize),
		numSubgraphs(aNumSamples),
		subgraphsPerSeedNode(aNumSamples / aGraphSize1),
		spatialTolerance(aSpatialTolerance),
		//inIntervals(inIntervals),
		//inNeighborIds(inNeighborIds),
		inNodeTypes(inTypes),
		inNodeIds(inIds),
		inBorderNodeFlags(inFlags),
		inDistMatrix(inMatrix),
		outNodeTypes(outTypes),
		outNodeIds(outIds),
		outBorderNodeFlags(outFlags),
		outDistMatrix(outMatrix),
		outValidSubgraphFlags(outSubgraphFlags)
	{}
	
	__host__ __device__ FORCE_INLINE void invalidateSubgraph(unsigned int subgraphStartLocation)
	{
		for (unsigned int localNodeId = 0u; localNodeId < subgraphSize; ++localNodeId)
		{
			outNodeIds[subgraphStartLocation + localNodeId] = graphSize2;
			outBorderNodeFlags[subgraphStartLocation + localNodeId] = 0u;
		}
	}

	__host__ __device__	void operator()(const size_t& aId_s)
	{
		unsigned int aId = (unsigned int)aId_s;
		unsigned int subgraphSeedNodeId = subgraphsPerSeedNode == 0u ? aId : aId / subgraphsPerSeedNode;
		unsigned int subgraphOffset = subgraphsPerSeedNode == 0u ? aId : aId % subgraphsPerSeedNode;
		unsigned int subgraphStartLocation = subgraphOffset * subgraphSize + subgraphSeedNodeId * subgraphsPerSeedNode * subgraphSize;
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
				outNodeIds[subgraphStartLocation + localNodeId] = 0u;
			}
			outBorderNodeFlags[subgraphStartLocation + localNodeId] = 0u;
		}
		
		if (interiorNodesCount == 0 || subgraphSize - interiorNodesCount < 3)
		{
			//invalid subgraph - too few nodes in the cut, or no interior nodes
			invalidateSubgraph(subgraphStartLocation);
			return;
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
					if (fabsf(currentDist - targetDist) > spatialTolerance)
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
						if (fabsf(currentDist - targetDist) < spatialTolerance)
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
					invalidateSubgraph(subgraphStartLocation);
					return;
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
				if (fabsf(currentDist - targetDist) > spatialTolerance)
				{
					foundMismatch = true;
					break;
				}
			}//end for each other node in the cut
		}//end for each node in the cut
		
		if (foundMismatch)
		{
			invalidateSubgraph(subgraphStartLocation);
		}
		outValidSubgraphFlags[aId] = 1u;

	}

};

class ValidSubgraphCompactor
{
public:
	unsigned int graphSize;
	unsigned int subgraphSize;
	unsigned int numSubgraphs;
	unsigned int subgraphsPerSeedNode;

	thrust::device_ptr<unsigned int> nodeIds1;
	thrust::device_ptr<unsigned int> nodeFlags1;
	thrust::device_ptr<unsigned int> nodeIds2;
	thrust::device_ptr<unsigned int> nodeFlags2;

	thrust::device_ptr<unsigned int> validSubgraphFlags;

	thrust::device_ptr<unsigned int> nodeIds3;
	thrust::device_ptr<unsigned int> nodeFlags3;
	thrust::device_ptr<unsigned int> nodeIds4;
	thrust::device_ptr<unsigned int> nodeFlags4;


	ValidSubgraphCompactor(
		unsigned int aGraphSize,
		unsigned int aSampleSize,
		unsigned int aNumSamples,
		thrust::device_ptr<unsigned int> aNodeIds1,
		thrust::device_ptr<unsigned int> aNodeFlags1,
		thrust::device_ptr<unsigned int> aNodeIds2,
		thrust::device_ptr<unsigned int> aNodeFlags2,
		thrust::device_ptr<unsigned int> aNodeIds3,
		thrust::device_ptr<unsigned int> aNodeFlags3,
		thrust::device_ptr<unsigned int> aNodeIds4,
		thrust::device_ptr<unsigned int> aNodeFlags4,
		thrust::device_ptr<unsigned int> outValidFlags
	) : graphSize(aGraphSize),
		subgraphSize(aSampleSize),
		numSubgraphs(aNumSamples),
		subgraphsPerSeedNode(aNumSamples / aGraphSize),
		nodeIds1(aNodeIds1),
		nodeFlags1(aNodeFlags1),
		nodeIds2(aNodeIds2),
		nodeFlags2(aNodeFlags2),
		nodeIds3(aNodeIds3),
		nodeFlags3(aNodeFlags3),
		nodeIds4(aNodeIds4),
		nodeFlags4(aNodeFlags4),
		validSubgraphFlags(outValidFlags)
	{}

	__host__ __device__	void operator()(const size_t& aId_s)
	{
		unsigned int aId = (unsigned int)aId_s;

		if (validSubgraphFlags[aId] == validSubgraphFlags[aId+1])
			return;

		unsigned int subgraphSeedNodeId = subgraphsPerSeedNode == 0u ? aId : aId / subgraphsPerSeedNode;
		unsigned int subgraphOffset = subgraphsPerSeedNode == 0u ? aId : aId % subgraphsPerSeedNode;
		unsigned int subgraphStartLocation = subgraphOffset * subgraphSize + subgraphSeedNodeId * subgraphsPerSeedNode * subgraphSize;

		unsigned int outputId = validSubgraphFlags[aId];
		unsigned int outSubgraphSeedNodeId = subgraphsPerSeedNode == 0u ? outputId : outputId / subgraphsPerSeedNode;
		unsigned int outSubgraphOffset = subgraphsPerSeedNode == 0u ? outputId : outputId % subgraphsPerSeedNode;
		unsigned int outSubgraphStartLocation = outSubgraphOffset * subgraphSize + outSubgraphSeedNodeId * subgraphsPerSeedNode * subgraphSize;


		for (unsigned int localNodeId = 0u; localNodeId < subgraphSize; ++localNodeId)
		{
			nodeIds3[outSubgraphStartLocation + localNodeId] = nodeIds1[subgraphStartLocation + localNodeId];
		}

		for (unsigned int localNodeId = 0u; localNodeId < subgraphSize; ++localNodeId)
		{
			nodeIds4[outSubgraphStartLocation + localNodeId] = nodeIds2[subgraphStartLocation + localNodeId];
		}

		for (unsigned int localNodeId = 0u; localNodeId < subgraphSize; ++localNodeId)
		{
			nodeFlags3[outSubgraphStartLocation + localNodeId] = nodeFlags1[subgraphStartLocation + localNodeId];
		}

		for (unsigned int localNodeId = 0u; localNodeId < subgraphSize; ++localNodeId)
		{
			nodeFlags4[outSubgraphStartLocation + localNodeId] = nodeFlags2[subgraphStartLocation + localNodeId];
		}

	}

};

__host__ Graph VariationGenerator::mergeGraphs(
	const Graph & aGraph1,
	const Graph & aGraph2,
	const thrust::host_vector<unsigned int>& aSubgraphFlags1,
	const thrust::host_vector<unsigned int>& aSubgraphFlags2,
	const thrust::host_vector<unsigned int>::iterator aSubgraphSeam1Begin,
	const thrust::host_vector<unsigned int>::iterator aSubgraphSeam2Begin,
	const thrust::host_vector<unsigned int>::iterator aSubgraphSeamFlags12Begin,
	const size_t aSeamSize)
{
	unsigned int nodeCount1 = thrust::reduce(aSubgraphFlags1.begin(), aSubgraphFlags1.end(), 0u, thrust::plus<unsigned int>());
	unsigned int nodeCount2 = thrust::reduce(aSubgraphFlags2.begin(), aSubgraphFlags2.end(), 0u, thrust::plus<unsigned int>());

	thrust::host_vector<unsigned int> intervals1Host(aGraph1.intervals);
	thrust::host_vector<unsigned int> intervals2Host(aGraph2.intervals);
	thrust::host_vector<unsigned int> intervalsHost(nodeCount1 + nodeCount2 + 1);
	intervalsHost[0] = 0u;

	thrust::host_vector<unsigned int> nodeIdMap1(aGraph1.numNodes(), nodeCount1 + nodeCount2);
	unsigned int currentId = 0u;
	for (size_t nodeId = 0; nodeId < aGraph1.numNodes(); ++nodeId)
	{
		if (aSubgraphFlags1[nodeId] == 1u)
		{
			nodeIdMap1[nodeId] = currentId++;
			unsigned int numEdges = intervals1Host[nodeId + 1] - intervals1Host[nodeId];
			intervalsHost[currentId] = intervalsHost[currentId - 1] + numEdges;
		}
		else
		{
			nodeIdMap1[nodeId] = nodeCount1 + nodeCount2;
		}
	}
	thrust::host_vector<unsigned int> nodeIdMap2(aGraph2.numNodes(), nodeCount1 + nodeCount2);
	for (size_t nodeId = 0; nodeId < aGraph2.numNodes(); ++nodeId)
	{
		if (aSubgraphFlags2[nodeId] == 1u)
		{
			nodeIdMap2[nodeId] = currentId++;
			unsigned int numEdges = intervals2Host[nodeId + 1] - intervals2Host[nodeId];
			intervalsHost[currentId] = intervalsHost[currentId - 1] + numEdges;
		}
		else
		{
			nodeIdMap2[nodeId] = nodeCount1 + nodeCount2;
		}
	}

	for (size_t subnodeId = 0; subnodeId < aSeamSize; ++subnodeId)
	{
		if (*(aSubgraphSeamFlags12Begin + subnodeId) == 1u)
		{
			nodeIdMap1[*(aSubgraphSeam1Begin + subnodeId)] = nodeIdMap2[*(aSubgraphSeam2Begin + subnodeId)];
		}
	}
	for (size_t subnodeId = 0; subnodeId < aSeamSize; ++subnodeId)
	{
		if (*(aSubgraphSeamFlags12Begin + subnodeId) == 2u)
			nodeIdMap2[*(aSubgraphSeam2Begin + subnodeId)] = nodeIdMap1[*(aSubgraphSeam1Begin + subnodeId)];		
	}

//#ifdef _DEBUG
//	outputHostVector("Node id map 1: ", nodeIdMap1);
//	outputHostVector("Node id map 2: ", nodeIdMap2);
//	thrust::host_vector<unsigned int> aSubgraphSeamFlags12(aSubgraphSeamFlags12Begin, aSubgraphSeamFlags12Begin + aSeamSize);
//	thrust::host_vector<unsigned int> aSubgraphSeamNodes1(aSubgraphSeam1Begin, aSubgraphSeam1Begin + aSeamSize);
//	thrust::host_vector<unsigned int> aSubgraphSeamNodes2(aSubgraphSeam2Begin, aSubgraphSeam2Begin + aSeamSize);
//	outputHostVector("Node seam ids 1: ", aSubgraphSeamNodes1);
//	outputHostVector("Node seam flags: ", aSubgraphSeamFlags12);
//	outputHostVector("Node seam ids 2: ", aSubgraphSeamNodes2);
//#endif

	thrust::host_vector<unsigned int> adjacencyVals1Host(aGraph1.adjacencyVals);
	thrust::host_vector<unsigned int> adjacencyVals2Host(aGraph2.adjacencyVals);

	thrust::host_vector<unsigned int> adjacencyKeysHost(intervalsHost[currentId]);
	thrust::host_vector<unsigned int> adjacencyValsHost(intervalsHost[currentId]);

	for (size_t nodeId = 0; nodeId < aGraph2.numNodes(); ++nodeId)
	{
		if (aSubgraphFlags2[nodeId] == 1u)
		{
			for (unsigned int edgeId = intervals2Host[nodeId]; edgeId < intervals2Host[nodeId + 1]; ++edgeId)
			{
				unsigned int outKey = nodeIdMap2[nodeId];
				unsigned int outIntervalBegin = intervalsHost[outKey];

				unsigned int localEdgeId = edgeId - intervals2Host[nodeId];
				unsigned int outEdgeId = outIntervalBegin + localEdgeId;

				adjacencyKeysHost[outEdgeId] = outKey;

				unsigned int inVal = adjacencyVals2Host[edgeId];
				unsigned int outVal = nodeIdMap2[inVal];

				adjacencyValsHost[outEdgeId] = outVal;
			}
		}
	}

	for (size_t nodeId = 0; nodeId < aGraph1.numNodes(); ++nodeId)
	{
		if (aSubgraphFlags1[nodeId] == 1u)
		{
			for (unsigned int edgeId = intervals1Host[nodeId]; edgeId < intervals1Host[nodeId + 1]; ++edgeId)
			{
				unsigned int outKey = nodeIdMap1[nodeId];
				unsigned int outIntervalBegin = intervalsHost[outKey];
				
				unsigned int localEdgeId = edgeId - intervals1Host[nodeId];
				unsigned int outEdgeId = outIntervalBegin + localEdgeId;
	
				adjacencyKeysHost[outEdgeId] = outKey;
				
				unsigned int inVal = adjacencyVals1Host[edgeId];
				unsigned int outVal = nodeIdMap1[inVal];

				adjacencyValsHost[outEdgeId] = outVal;
			}
		}
	}

	//handle dangling edges in subgraph 1
	for (size_t seamNodeId = 0; seamNodeId < aSeamSize; ++seamNodeId)
	{
		if (*(aSubgraphSeamFlags12Begin + seamNodeId) == 2u)
		{
			unsigned int nodeId1 = *(aSubgraphSeam1Begin + seamNodeId);
			unsigned int nodeId2 = *(aSubgraphSeam2Begin + seamNodeId);
			unsigned int outNodeId1 = nodeIdMap1[nodeId1];
			//find all neigbors of the "outside" node (second graph)
			for (unsigned int edgeId = intervalsHost[outNodeId1]; edgeId < intervalsHost[outNodeId1 + 1]; ++edgeId)
			{
				unsigned int outVal = adjacencyValsHost[edgeId];
				if (outVal == nodeCount1 + nodeCount2)
				{
					for (unsigned int edgeId2 = intervals2Host[nodeId2]; edgeId2 < intervals2Host[nodeId2 + 1] && outVal == nodeCount1 + nodeCount2; ++edgeId2)
					{
						unsigned int inVal2 = adjacencyVals2Host[edgeId2];
						unsigned int outVal2 = nodeIdMap2[inVal2];
						//check if already connected
						bool connected = false;
						for (unsigned int edgeIdOut = intervalsHost[outNodeId1]; edgeIdOut < intervalsHost[outNodeId1 + 1] && !connected; ++edgeIdOut)
						{
							if (adjacencyValsHost[edgeIdOut] == outVal2)
								connected = true;
						}
						if (!connected)
						{
							outVal = outVal2;
							adjacencyValsHost[edgeId] = outVal2;
							break;
						}
					}
				}
			}
		}
	}

	Graph result;
	result.intervals = thrust::device_vector<unsigned int>(intervalsHost);
	result.adjacencyKeys = thrust::device_vector<unsigned int>(adjacencyKeysHost);
	result.adjacencyVals = thrust::device_vector<unsigned int>(adjacencyValsHost);
	
//#ifdef _DEBUG
//	outputDeviceVector("merged graph intervals: ", result.intervals);
//	outputDeviceVector("merged edge keys: ", result.adjacencyKeys);
//	outputDeviceVector("merged edge vals: ", result.adjacencyVals);
//#endif

	return result;
}

__host__ std::string VariationGenerator::operator()(const char * aFilePath1, const char * aFilePath2,
	WFObject & aObj1, WFObject & aObj2, Graph & aGraph1, Graph & aGraph2, float aRelativeThreshold)
{
	cudastd::timer timer;
	cudastd::timer intermTimer;

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
	for (size_t nodeId = 0; nodeId < aObj2.objects.size(); ++nodeId)
	{
		size_t faceId = aObj2.objects[nodeId].x;
		size_t materialId = aObj2.faces[faceId].material;
		nodeTypes2Host[nodeId] = (unsigned int)materialId;
	}
	thrust::device_vector<unsigned int> nodeTypes2(nodeTypes2Host);

	initTime = intermTimer.get();
	intermTimer.start();

	const unsigned int numSubgraphSamples = (unsigned int)objCenters1.size();
	const unsigned int subgraphSampleSize = (unsigned int)objCenters1.size() / 2u;

	if (subgraphSampleSize < 3)
		return "";

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

	samplingTime = intermTimer.get();
	intermTimer.start();

//#ifdef _DEBUG
//	outputDeviceVector("Subgraph node ids     1: ", subgraphNodeIds1);
//	outputDeviceVector("Subgraph border flags 1: ", subgraphBorderFlags1);
//#endif

	float3 minBound, maxBound;
	ObjectBoundsExporter()(aObj1, minBound, maxBound);
	const float boundsDiagonal = len(maxBound - minBound);
	const float spatialTolerance = boundsDiagonal * 0.577350269f * aRelativeThreshold;

	thrust::device_vector<unsigned int> validSubgraphFlags(numSubgraphSamples + 1, 0u);

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
		nodeTypes2.data(),
		subgraphNodeIds2.data(),
		subgraphBorderFlags2.data(),
		pairwiseDistMatrix2.data(),
		validSubgraphFlags.data()
	);

	thrust::for_each(first, lastSubgraph, matchCuts);

	matchingTime = intermTimer.get();
	intermTimer.start();

//#ifdef _DEBUG
//	outputDeviceVector("Subgraph node ids     2: ", subgraphNodeIds2);
//	outputDeviceVector("Subgraph border flags 2: ", subgraphBorderFlags2);
//	outputDeviceVector("Valid subgraph flags   : ", validSubgraphFlags);
//#endif

	thrust::device_vector<unsigned int> subgraphComplementNodeIds(numSubgraphSamples * objCenters2.size());

	thrust::exclusive_scan(validSubgraphFlags.begin(), validSubgraphFlags.end(), validSubgraphFlags.begin());
	size_t numValidSubgraphs = validSubgraphFlags[validSubgraphFlags.size() - 1];

	thrust::device_vector<unsigned int> subgraphNodeIds3(numValidSubgraphs * subgraphSampleSize);
	thrust::device_vector<unsigned int> subgraphBorderFlags3(numValidSubgraphs * subgraphSampleSize);

	thrust::device_vector<unsigned int> subgraphNodeIds4(numValidSubgraphs * subgraphSampleSize);
	thrust::device_vector<unsigned int> subgraphBorderFlags4(numValidSubgraphs * subgraphSampleSize);

	ValidSubgraphCompactor compactValid(
		(unsigned int)objCenters1.size(),
		subgraphSampleSize,
		numSubgraphSamples,
		subgraphNodeIds1.data(),
		subgraphBorderFlags1.data(),
		subgraphNodeIds2.data(),
		subgraphBorderFlags2.data(),
		subgraphNodeIds3.data(),
		subgraphBorderFlags3.data(),
		subgraphNodeIds4.data(),
		subgraphBorderFlags4.data(),
		validSubgraphFlags.data()
	);
	thrust::for_each(first, lastSubgraph, compactValid);

	subgraphNodeIds1.clear();
	subgraphBorderFlags1.clear();
	subgraphNodeIds2.clear();
	subgraphBorderFlags2.clear();

	subgraphNodeIds1.shrink_to_fit();
	subgraphBorderFlags1.shrink_to_fit();
	subgraphNodeIds2.shrink_to_fit();
	subgraphBorderFlags2.shrink_to_fit();

	compactionTime = intermTimer.get();
	intermTimer.start();

	thrust::host_vector<unsigned int> subgraphNodeIdsHost1(subgraphNodeIds3);
	thrust::host_vector<unsigned int> subgraphBorderFlagsHost1(subgraphBorderFlags3);

	thrust::host_vector<unsigned int> subgraphNodeIdsHost2(subgraphNodeIds4);
	thrust::host_vector<unsigned int> subgraphBorderFlagsHost2(subgraphBorderFlags4);

	thrust::host_vector<unsigned int> graph2Intervals(aGraph2.intervals);
	thrust::host_vector<unsigned int> graph2NbrIds(aGraph2.adjacencyVals);

	unsigned int graphSize1 = (unsigned int)objCenters1.size();
	unsigned int graphSize2 = (unsigned int)objCenters2.size();

	std::string result = "";
	GraphToStringConverter convertToStr;
	CollisionGraphExporter graphExporter;
	numVariations = 0u;
	std::vector<NodeTypeHistogram> variatioHistograms;
	variatioHistograms.push_back(NodeTypeHistogram(nodeTypes1));
	variatioHistograms.push_back(NodeTypeHistogram(nodeTypes2));

	for (unsigned int id = 0u; id < numValidSubgraphs; ++id)
	{
		thrust::host_vector<unsigned int> completeSubgraphFlags2(graphSize2, 0u);
		std::vector<unsigned int> nodeStack;
		unsigned int subgraph2Size = 0u;
		unsigned int complementSize = 0u;
		thrust::host_vector<unsigned int>::iterator subgraphNodeIdsHost1Begin = subgraphNodeIdsHost1.begin() + id * subgraphSampleSize;
		thrust::host_vector<unsigned int>::iterator subgraphNodeIdsHost2Begin = subgraphNodeIdsHost2.begin() + id * subgraphSampleSize;
		thrust::host_vector<unsigned int>::iterator subgraphBorderFlagsHost1Begin = subgraphBorderFlagsHost1.begin() + id * subgraphSampleSize;
		thrust::host_vector<unsigned int>::iterator subgraphBorderFlagsHost2Begin = subgraphBorderFlagsHost2.begin() + id * subgraphSampleSize;

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

		//if (subgraph2Size + subgraph1Size == graphSize2 || subgraph2Size + subgraph1Size == graphSize1)
		//	continue;//TODO: accept variations with the same size but different node types

		///////////////////////////////////////////////////////////////////////////////////
		//discard variation with repeating node type histograms
		thrust::host_vector<unsigned int> nodeTypes(subgraph2Size + subgraph1Size);
		auto outTypeIt = nodeTypes.begin();
		for (auto inTypeIt1 = nodeTypes1.begin(); inTypeIt1 != nodeTypes1.end(); ++inTypeIt1)
		{
			if (completeSubgraphFlags1[inTypeIt1 - nodeTypes1.begin()] == 1u)
			{
				*outTypeIt = *inTypeIt1;
				outTypeIt++;
			}
		}
		for (auto inTypeIt2 = nodeTypes2.begin(); inTypeIt2 != nodeTypes2.end(); ++inTypeIt2)
		{
			if (completeSubgraphFlags2[inTypeIt2 - nodeTypes2.begin()] == 1u)
			{
				*outTypeIt = *inTypeIt2;
				outTypeIt++;
			}
		}
		NodeTypeHistogram typeHist(nodeTypes);
		bool repeatedHistogram = false;
		for (auto it = variatioHistograms.begin(); it != variatioHistograms.end() && !repeatedHistogram; ++it)
		{
			if (*it == typeHist)
				repeatedHistogram = true;
		}
		if (repeatedHistogram)
			continue;
		variatioHistograms.push_back(typeHist);
		////////////////////////////////////////////////////////////////////////////////////////

		for (unsigned int i = 0u; i < graphSize2; ++i)
		{
			if (completeSubgraphFlags2[i] == 2u)
				completeSubgraphFlags2[i] = 0u;
		}

		//graphExporter.exportSubGraph(aFilePath1, aObj1, aGraph1, numVariations, completeSubgraphFlags1);
		//graphExporter.exportSubGraph(aFilePath2, aObj2, aGraph2, numVariations, completeSubgraphFlags2);

		++numVariations;

		//Graph variationGraph = mergeGraphs(aGraph1, aGraph2,
		//	completeSubgraphFlags1, completeSubgraphFlags2,
		//	subgraphNodeIdsHost1Begin,
		//	subgraphNodeIdsHost2Begin,
		//	subgraphBorderFlagsHost1Begin,
		//	subgraphSampleSize);
		//
		//std::string graphStrings = convertToStr.toString(variationGraph, nodeTypes);
		//result.append(graphStrings);

		///////////////////////////////////////////////////////////////////////////////////
		//Find correspondence transformation between both sub-graphs

		//Compute the means of the border node locations
		float3 center1 = make_float3(0.f, 0.f, 0.f);
		float3 center2 = make_float3(0.f, 0.f, 0.f);
		float numPoints = 0.f;
		for (unsigned int i = 0u; i < subgraphSampleSize; ++i)
		{
			if (*(subgraphBorderFlagsHost2Begin + i) != 0u)
			{
				center1   += objCenters1[*(subgraphNodeIdsHost1Begin + i)];
				center2   += objCenters2[*(subgraphNodeIdsHost2Begin + i)];
				numPoints += 1.f;
			}
		}
		center1 /= numPoints;
		center2 /= numPoints;

		//Compute covariance matrix
		float covMat[3][3];
		for (unsigned int i = 0u; i < subgraphSampleSize; ++i)
		{
			if (*(subgraphBorderFlagsHost2Begin + i) != 0u)
			{
				float3 vec1 = objCenters1[*(subgraphNodeIdsHost1Begin + i)] - center1;
				float3 vec2 = objCenters2[*(subgraphNodeIdsHost2Begin + i)] - center2;

				covMat[0][0] += vec2.x * vec1.x;
				covMat[1][0] += vec2.y * vec1.x;
				covMat[2][0] += vec2.z * vec1.x;

				covMat[0][1] += vec2.x * vec1.y;
				covMat[1][1] += vec2.y * vec1.y;
				covMat[2][1] += vec2.z * vec1.y;

				covMat[0][2] += vec2.x * vec1.z;
				covMat[1][2] += vec2.y * vec1.z;
				covMat[2][2] += vec2.z * vec1.z;
			}
		}
		//Singular Value Decomposition
		float diag[3];
		float vMat[3][3];

		svdcmp(covMat, diag, vMat);

		//Rotation is V * transpose(U)
		float rotMat[3][3];
		for (unsigned int row = 0; row < 3; ++row)
		{
			for (unsigned int col = 0; col < 3; ++col)
			{
				rotMat[row][col] = vMat[row][0] * covMat[col][0] + vMat[row][1] * covMat[col][1] + vMat[row][2] * covMat[col][2];
			}
		}

		quaternion4f rotation(
			rotMat[0][0], rotMat[1][0], rotMat[2][0],
			rotMat[0][1], rotMat[1][1], rotMat[2][1],
			rotMat[0][2], rotMat[1][2], rotMat[2][2]
		);

		float rotDet = determinant(
			rotMat[0][0], rotMat[1][0], rotMat[2][0],
			rotMat[0][1], rotMat[1][1], rotMat[2][1],
			rotMat[0][2], rotMat[1][2], rotMat[2][2]
		);

		if (rotDet < 0.f || fabsf(rotDet - 1.f) > EPS)
			continue;
		///////////////////////////////////////////////////////////////////////////////////

		///////////////////////////////////////////////////////////////////////////////////
		//Create the variation by merging the subsets of aObj1 and aObj2
		WFObject variation = WFObjectMerger()(aObj1, center1, aObj2, center2, rotation, completeSubgraphFlags1, completeSubgraphFlags2);
		///////////////////////////////////////////////////////////////////////////////////

		///////////////////////////////////////////////////////////////////////////////////
		//Compute the collision graph for the variation
		CollisionDetector detector;
		Graph variationGraph = detector.computeCollisionGraph(variation, 0.02f);
		///////////////////////////////////////////////////////////////////////////////////

		///////////////////////////////////////////////////////////////////////////////////
		//Check that the variation graph is valid

		//TODO:Check that the variation graph is valid

		///////////////////////////////////////////////////////////////////////////////////

		std::string fileName1(aFilePath1);
		if (fileName1.find_last_of("/\\") == std::string::npos)
			fileName1 = fileName1.substr(0, fileName1.size() - 5);
		else
			fileName1 = fileName1.substr(fileName1.find_last_of("/\\") + 1, fileName1.size() - fileName1.find_last_of("/\\") - 5);

		std::string fileName2(aFilePath1);
		if (fileName2.find_last_of("/\\") == std::string::npos)
			fileName2 = fileName2.substr(0, fileName2.size() - 5);
		else
			fileName2 = fileName2.substr(fileName2.find_last_of("/\\") + 1, fileName2.size() - fileName2.find_last_of("/\\") - 5);


		std::string objDir = getDirName(aFilePath2);
		std::string variationFilePath = objDir + fileName1 + "_" + fileName2 + "_var_" + itoa((int)numVariations) + ".obj";

		graphExporter.exportCollisionGraph(variationFilePath.c_str(), variation, variationGraph);

		std::string variationStrings = convertToStr(variation, variationGraph);
		result.append(variationStrings);
		//std::string result = convertToStr(variation, variationGraph);

	
	}
	extractionTime = intermTimer.get();

	totalTime = timer.get();

	intermTimer.cleanup();
	timer.cleanup();

	return result;
}


__host__ void VariationGenerator::stats()
{
	std::cerr << "Created "<< numVariations <<" variations in " << totalTime << "ms\n";
	std::cerr << "Initialization in      " << initTime << "ms\n";
	std::cerr << "Subgraph sampling in   " << samplingTime << "ms\n";
	std::cerr << "Graph cut matching in  " << matchingTime << "ms\n";
	std::cerr << "Compaction in          " << compactionTime << "ms\n";
	std::cerr << "Extraction in          " << extractionTime << "ms\n";
}