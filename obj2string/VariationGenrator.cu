#include "pch.h"
#include "VariationGenerator.h"

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/scan.h>

#include "Algebra.h"
#include "WFObjUtils.h"

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

				outNodeIds[subgraphStartLocation + localNodeId] += 1;
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

class ValidSubgraphMarker
{
public:
	unsigned int graphSize1;
	unsigned int graphSize2;
	unsigned int subgraphSize;
	unsigned int numSubgraphs;
	unsigned int subgraphsPerSeedNode;

	thrust::device_ptr<unsigned int> nodeIds1;
	thrust::device_ptr<unsigned int> nodeFlags1;
	thrust::device_ptr<unsigned int> nodeIds2;
	thrust::device_ptr<unsigned int> nodeFlags2;

	thrust::device_ptr<unsigned int> outValidSubgraphFlags;

	ValidSubgraphMarker(
		unsigned int aGraphSize1,
		unsigned int aGraphSize2,
		unsigned int aSampleSize,
		unsigned int aNumSamples,
		thrust::device_ptr<unsigned int> aNodeIds1,
		thrust::device_ptr<unsigned int> aNodeFlags1,
		thrust::device_ptr<unsigned int> aNodeIds2,
		thrust::device_ptr<unsigned int> aNodeFlags2,
		thrust::device_ptr<unsigned int> outValidFlags
	) : graphSize1(aGraphSize1),
		graphSize2(aGraphSize2),
		subgraphSize(aSampleSize),
		numSubgraphs(aNumSamples),
		subgraphsPerSeedNode(aNumSamples / aGraphSize1),
		nodeIds1(aNodeIds1),
		nodeFlags1(aNodeFlags1),
		nodeIds2(aNodeIds2),
		nodeFlags2(aNodeFlags2),
		outValidSubgraphFlags(outValidFlags)
	{}

	__host__ __device__	void operator()(const size_t& aId_s)
	{
		unsigned int aId = (unsigned int)aId_s;

		unsigned int subgraphSeedNodeId = subgraphsPerSeedNode == 0u ? aId : aId / subgraphsPerSeedNode;
		unsigned int subgraphOffset = subgraphsPerSeedNode == 0u ? aId : aId % subgraphsPerSeedNode;
		unsigned int subgraphStartLocation = subgraphOffset * subgraphSize + subgraphSeedNodeId * subgraphsPerSeedNode * subgraphSize;

		for (unsigned int localNodeId = 0u; localNodeId < subgraphSize; ++localNodeId)
		{
			if (nodeIds2[subgraphStartLocation + localNodeId] != graphSize2)
			{
				outValidSubgraphFlags[aId] = 1;
				return;
			}
		}
		outValidSubgraphFlags[aId] = 0;
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

__host__ std::string VariationGenerator::operator()(const char * aFilePath, WFObject & aObj1, WFObject & aObj2, Graph & aGraph1, Graph & aGraph2, float aRelativeThreshold)
{
	cudastd::timer timer;
	cudastd::timer intermTimer;

	std::vector<float3> objCenters1;
	std::vector<float> objSizes1;

	ObjectCenterExporter()(aObj1, objCenters1, objSizes1, 0.3333f);

	std::vector<float3> objCenters2;
	std::vector<float> objSizes2;

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

	const unsigned int numSubgraphSamples = 2;// (unsigned int)objCenters1.size();
	const unsigned int subgraphSampleSize = 11;// (unsigned int)objCenters1.size() / 2u;

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

#ifdef _DEBUG
	outputDeviceVector("Subgraph node ids     1: ", subgraphNodeIds1);
	outputDeviceVector("Subgraph border flags 1: ", subgraphBorderFlags1);
#endif

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

#ifdef _DEBUG
	outputDeviceVector("Subgraph node ids     2: ", subgraphNodeIds2);
	outputDeviceVector("Subgraph border flags 2: ", subgraphBorderFlags2);
	outputDeviceVector("Valid subgraph flags   : ", validSubgraphFlags);
#endif

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

	compactionTime = intermTimer.get();
	intermTimer.start();


	numVariations = numValidSubgraphs;
	totalTime = timer.get();

	intermTimer.cleanup();
	timer.cleanup();

	return "";
}


__host__ void VariationGenerator::stats()
{
	std::cerr << "Created "<< numVariations <<" variations in " << totalTime << "ms\n";
	std::cerr << "Initialization in      " << initTime << "ms\n";
	std::cerr << "Subgraph sampling in   " << samplingTime << "ms\n";
	std::cerr << "Graph cut matching in  " << matchingTime << "ms\n";
	std::cerr << "Compaction in          " << compactionTime << "ms\n";
}