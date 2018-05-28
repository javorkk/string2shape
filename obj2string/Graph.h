#ifdef _MSC_VER
#pragma once
#endif

#ifndef GRAPH_H_2FBC6906_D571_4BD5_AAD5_86D60FAC5013
#define GRAPH_H_2FBC6906_D571_4BD5_AAD5_86D60FAC5013

#include <thrust/device_vector.h>
#include "Algebra.h"

class Graph
{
public:
	//per node indirections into the edge/adjacency list
	thrust::device_vector<unsigned int> intervals;
	//edges stored as adjacency list (key-value pairs)
	thrust::device_vector<unsigned int> adjacencyKeys;
	thrust::device_vector<unsigned int> adjacencyVals;
	//edge types : 0 not connected, 1 spanning tree, 2 cycle
	enum EdgeType {NOT_CONNECTED = 0, SPANNING_TREE = 1, CYCLE = 2};

	__host__ FORCE_INLINE size_t numNodes() const { return intervals.size() > 1u ? intervals.size() - 1u : 0u; }

	__host__ FORCE_INLINE size_t numEdges() const { return adjacencyVals.size() / 2u; }

	__host__ FORCE_INLINE unsigned int neighborsBegin(const unsigned int aNodeId) const
	{
		if (aNodeId >= intervals.size() - 2)
			return (unsigned int) -1;//invalid node id

		return intervals[aNodeId];
	}
	__host__ FORCE_INLINE unsigned int neighborsEnd(const unsigned int aNodeId) const
	{
		if (aNodeId >= intervals.size() - 2)
			return (unsigned int)-1;//invalid node id

		return intervals[aNodeId + 1];
	}
	__host__ FORCE_INLINE unsigned int getNeighbor(const unsigned int aNeighborId) const
	{
		if (aNeighborId >= adjacencyVals.size() - 1)
			return (unsigned int)-1;//invalid id

		return adjacencyVals[aNeighborId];
	}

	__host__ FORCE_INLINE unsigned int getEdgeId(unsigned int NodeIdA, unsigned int NodeIdB) const
	{
		if (NodeIdA >= numNodes() || NodeIdB >= numNodes())
			return (unsigned)numEdges() * 2u;
		thrust::host_vector<unsigned int> hostVals(adjacencyVals.begin() + intervals[NodeIdA], adjacencyVals.begin() + intervals[NodeIdA + 1]);
		for (unsigned int id = 0u; id < hostVals.size(); ++id)
			if (hostVals[id] == NodeIdB)
				return id + intervals[NodeIdA];
		return (unsigned)numEdges() * 2u;
	}

	__host__ FORCE_INLINE unsigned int getOpositeEdgeId(unsigned int aEdgeId) const
	{
		if (aEdgeId >= (unsigned)numEdges() * 2u)
			return (unsigned)numEdges() * 2u;
		unsigned int oposingKey = adjacencyVals[aEdgeId];
		unsigned int oposingVal = adjacencyKeys[aEdgeId];
		return  getEdgeId(oposingKey, oposingVal);
	}

	
	__host__ void fromAdjacencyMatrix(thrust::device_vector<unsigned int>& aAdjacencyMatrix, size_t aStride);
	
	__host__ void toAdjacencyMatrix(thrust::device_vector<unsigned int>& oAdjacencyMatrix, size_t& oStride);
	
	__host__ void toSpanningTree(thrust::device_vector<EdgeType>& oAdjacencyMatrix, size_t& oStride);

	//assumes that the lists are copied in adjacencyKeys and adjacencyVals
	__host__ void fromAdjacencyList(size_t aNumNodes, bool aSorted = true);

	__host__ int testGraphConstruction(int aGraphSize);

	__host__ int testSpanningTreeConstruction();

	__host__ bool isConnected();
};

#endif // GRAPH_H_2FBC6906_D571_4BD5_AAD5_86D60FAC5013
