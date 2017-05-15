#include "pch.h"
#include "Graph.h"

#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/scan.h>
#include <thrust/transform_scan.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include "DebugUtils.h"
#include "Timer.h"

struct isEdge
{ 
	__host__ __device__ unsigned int operator()(const unsigned int& x) const
	{ 
		return x != 0 ? 1u : 0u; 
	} 
};

class EdgeListWriter
{
public:
	size_t stride;
	thrust::device_ptr<unsigned int> keys;
	thrust::device_ptr<unsigned int> vals;


	EdgeListWriter(
		size_t aStride,
		thrust::device_ptr<unsigned int> aKeys,
		thrust::device_ptr<unsigned int> aVals
	):stride(aStride), keys(aKeys), vals(aVals)
	{}

	template <typename Tuple>
	__host__ __device__	void operator()(Tuple t)
	{
		const unsigned int myValue = thrust::get<0>(t);
		const unsigned int myOutPosition = thrust::get<1>(t);
		const size_t myColId = thrust::get<2>(t) % stride;
		const size_t myRowId = thrust::get<2>(t) / stride;
		if (myValue != 0u)
		{
			keys[myOutPosition] = (unsigned int)myRowId;
			vals[myOutPosition] = (unsigned int)myColId;
		}

	}

};

class EdgeMatrixWriter
{
public:
	size_t stride;
	thrust::device_ptr<unsigned int> matrix;


	EdgeMatrixWriter(
		size_t aStride,
		thrust::device_ptr<unsigned int> aMatrix
	) :stride(aStride), matrix(aMatrix)
	{}

	template <typename Tuple>
	__host__ __device__	void operator()(Tuple t)
	{
		const unsigned int myRowId = thrust::get<0>(t);
		const unsigned int myColId = thrust::get<1>(t);
		matrix[myColId + myRowId * stride] = 1u;
	}

};

class EdgeIdSelector
{
public:
	unsigned int stride;
	thrust::device_ptr<unsigned int> bestEdge;
	thrust::device_ptr<unsigned int> adjIntervals;
	thrust::device_ptr<unsigned int> neighborIds;
	thrust::device_ptr<unsigned int> superNodeIds;

	EdgeIdSelector(
		size_t aStride,
		thrust::device_ptr<unsigned int> aBestEdge,
		thrust::device_ptr<unsigned int> aIntervals,
		thrust::device_ptr<unsigned int> aNeighbors,
		thrust::device_ptr<unsigned int> aSuperNodeIds
	) :stride((unsigned int)aStride), bestEdge(aBestEdge),
		adjIntervals(aIntervals), neighborIds(aNeighbors),
		superNodeIds(aSuperNodeIds)
	{}

	__host__ __device__	void operator()(const size_t& aNodeId)
	{
		const unsigned int myNodeId = (unsigned int)aNodeId;
		const unsigned int mySuperNodeId = superNodeIds[myNodeId];
		unsigned int nbrSuperNodeId = mySuperNodeId;
		unsigned int bestEdgeId = (unsigned int)-1;
		for (unsigned int edgeId = adjIntervals[myNodeId]; edgeId < adjIntervals[myNodeId + 1]; ++edgeId)
		{
			const unsigned int nbrNodeId = neighborIds[edgeId];
			const unsigned int tmpSuperNodeId = superNodeIds[nbrNodeId];
			if (tmpSuperNodeId < nbrSuperNodeId)
			{
				nbrSuperNodeId = tmpSuperNodeId;
				bestEdgeId = edgeId;
			}
		}
		if (mySuperNodeId <= nbrSuperNodeId)
			return;
		//will overwrite alternative edges
		bestEdge[mySuperNodeId] = bestEdgeId;
		superNodeIds[stride] = 1u;

	}
};

class SuperNodeIdSetter
{
public:
	unsigned int numEdges;

	thrust::device_ptr<unsigned int> superNodeIds;
	thrust::device_ptr<unsigned int> neighborIds;
	thrust::device_ptr<unsigned int> bestEdge;
	
	thrust::device_ptr<unsigned int> edgeFlags;

	SuperNodeIdSetter(
		size_t aNumEdges,
		thrust::device_ptr<unsigned int> aSuperNodeIds,
		thrust::device_ptr<unsigned int> aNeighbors,
		thrust::device_ptr<unsigned int> aBestEdge,
		thrust::device_ptr<unsigned int> aEdgeFlags
		) :
		numEdges((unsigned int)aNumEdges),
		superNodeIds(aSuperNodeIds),
		neighborIds(aNeighbors),
		bestEdge(aBestEdge),
		edgeFlags(aEdgeFlags)
	{}

	__host__ __device__	void operator()(const size_t& aNodeId)
	{
		const unsigned int initialSuperNodeId = superNodeIds[aNodeId];
		unsigned int mySuperNodeId = initialSuperNodeId;
		const unsigned int initialEdgeId = bestEdge[mySuperNodeId];
		const unsigned int otherNodeId = initialEdgeId >= numEdges ? (unsigned int)aNodeId : neighborIds[initialEdgeId];
		unsigned int itsSuperNodeId = superNodeIds[otherNodeId];
		while (mySuperNodeId != itsSuperNodeId)
		{
			mySuperNodeId = itsSuperNodeId;
			const unsigned int nextEdgeId = bestEdge[itsSuperNodeId];
			const unsigned int nextNodeId = nextEdgeId >= numEdges ? itsSuperNodeId : neighborIds[nextEdgeId];
			itsSuperNodeId = superNodeIds[nextNodeId];
		}

		if (initialEdgeId < numEdges && mySuperNodeId != initialSuperNodeId)
		{
			superNodeIds[aNodeId] = mySuperNodeId;
			edgeFlags[initialEdgeId] = 1u;
		}
	}
};

class SpanningTreeEdgeInitializer
{
public:
	size_t stride;
	thrust::device_ptr<Graph::EdgeType> matrix;


	SpanningTreeEdgeInitializer(
		size_t aStride,
		thrust::device_ptr<Graph::EdgeType> aMatrix
	) :stride(aStride), matrix(aMatrix)
	{}

	template <typename Tuple>
	__host__ __device__	void operator()(Tuple t)
	{
		const unsigned int myNodeId0 = thrust::get<0>(t);
		const unsigned int myNodeId1 = thrust::get<1>(t);
		const unsigned int edgeFlag = thrust::get<2>(t);

		matrix[myNodeId0 + myNodeId1 * stride] = Graph::EdgeType::CYCLE;
		matrix[myNodeId1 + myNodeId0 * stride] = Graph::EdgeType::CYCLE;
	}

};

class SpanningTreeEdgeWriter
{
public:
	size_t stride;
	thrust::device_ptr<Graph::EdgeType> matrix;


	SpanningTreeEdgeWriter(
		size_t aStride,
		thrust::device_ptr<Graph::EdgeType> aMatrix
	) :stride(aStride), matrix(aMatrix)
	{}

	template <typename Tuple>
	__host__ __device__	void operator()(Tuple t)
	{
		const unsigned int myNodeId0 = thrust::get<0>(t);
		const unsigned int myNodeId1 = thrust::get<1>(t);
		const unsigned int edgeFlag  = thrust::get<2>(t);

		if (edgeFlag != 0)
		{
			matrix[myNodeId0 + myNodeId1 * stride] = Graph::EdgeType::SPANNING_TREE;
			matrix[myNodeId1 + myNodeId0 * stride] = Graph::EdgeType::SPANNING_TREE;
		}
	}

};

class IntervalExtractor
{
public:
	thrust::device_ptr<unsigned int> intervals;

	IntervalExtractor(thrust::device_ptr<unsigned int> aIntervals):intervals(aIntervals)
	{}

	template <typename Tuple>
	__host__ __device__	void operator()(Tuple t)
	{
		const unsigned int myNodeIndex = thrust::get<0>(t);
		const unsigned int nextNodeIndex = thrust::get<1>(t);
		const size_t myId = thrust::get<2>(t);
		for (unsigned int nodeIndex = myNodeIndex + 1; nodeIndex <= nextNodeIndex; ++nodeIndex)
		{
			intervals[nodeIndex] = (unsigned int)myId + 1u;
		}
	}

};

class MatrixRowCopy
{
	size_t stride;
	thrust::device_ptr<unsigned int> matPrefix;
	thrust::device_ptr<unsigned int> intervals;
public:
	MatrixRowCopy(
		size_t aStride,
		thrust::device_ptr<unsigned int> aPrefix,
		thrust::device_ptr<unsigned int> aIntervals
	) : stride(aStride), matPrefix(aPrefix), intervals(aIntervals)
	{}

	__host__ __device__	void operator()(size_t aRowId)
	{
		intervals[aRowId] = matPrefix[stride * aRowId];
	}

};


__host__ void Graph::fromAdjacencyMatrix(thrust::device_vector<unsigned int>& aAdjacencyMatrix, size_t aStride)
{
	if (aAdjacencyMatrix.size() != aStride * aStride)
		return; //something is wrong


	thrust::device_vector<unsigned int> matrixPrefix(aAdjacencyMatrix.size() + 1);
	thrust::transform(aAdjacencyMatrix.begin(), aAdjacencyMatrix.end(), matrixPrefix.begin(), isEdge());
	thrust::exclusive_scan(matrixPrefix.begin(), matrixPrefix.end(), matrixPrefix.begin());

	intervals = thrust::device_vector<unsigned int>(aStride + 1);
	
	thrust::counting_iterator<size_t> firstRowId(0u);
	thrust::counting_iterator<size_t> lastRowIdP1(aStride + 1u);
	MatrixRowCopy copyOp(aStride, matrixPrefix.data(), intervals.data());
	thrust::for_each(firstRowId, lastRowIdP1, copyOp);

	unsigned int numEdges = matrixPrefix[aAdjacencyMatrix.size()];

	adjacencyKeys = thrust::device_vector<unsigned int>(numEdges);
	adjacencyVals = thrust::device_vector<unsigned int>(numEdges);

	thrust::counting_iterator<size_t> first(0u);
	thrust::counting_iterator<size_t> last(aAdjacencyMatrix.size());

	EdgeListWriter writeEdges(aStride, adjacencyKeys.data(), adjacencyVals.data());

	thrust::for_each(
		thrust::make_zip_iterator(thrust::make_tuple(aAdjacencyMatrix.begin(), matrixPrefix.begin(), first)),
		thrust::make_zip_iterator(thrust::make_tuple(aAdjacencyMatrix.end(), matrixPrefix.end(), last)),
		writeEdges);

//#ifdef _DEBUG
//	outputDeviceVector("Edge X: ", adjacencyKeys);
//	outputDeviceVector("Edge Y: ", adjacencyVals);
//	outputDeviceVector("Extracted intervals: ", intervals);
//#endif
}

__host__ void Graph::toAdjacencyMatrix(thrust::device_vector<unsigned int>& oAdjacencyMatrix, size_t & oStride)
{
	if (intervals.size() < 2u)
		return; //emtpy graph

	oStride = intervals.size() - 1u;
	oAdjacencyMatrix = thrust::device_vector<unsigned int>(oStride * oStride);

	EdgeMatrixWriter writeEdges(oStride, oAdjacencyMatrix.data());

	thrust::for_each(
		thrust::make_zip_iterator(thrust::make_tuple(adjacencyKeys.begin(), adjacencyVals.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(adjacencyKeys.end(), adjacencyVals.end())),
		writeEdges);
}

__host__ void Graph::toSpanningTree(thrust::device_vector<EdgeType>& oAdjacencyMatrix, size_t & oStride)
{
	if (intervals.size() < 2u)
		return; //emtpy graph

	oStride = intervals.size() - 1u;
	const unsigned int numEdges = (unsigned int)adjacencyVals.size();
	thrust::device_vector<unsigned int> edgeFlags(numEdges, 0u);
	thrust::device_vector<unsigned int> superNodeIds(oStride + 1u); //last element is a termination flag
	thrust::sequence(superNodeIds.begin(), superNodeIds.end(), 0u);

	thrust::device_vector<unsigned int> bestEdge(oStride, numEdges);
	EdgeIdSelector edgeVote(oStride, bestEdge.data(), intervals.data(), adjacencyVals.data(), superNodeIds.data());
	SuperNodeIdSetter updateSuperNodes(numEdges, superNodeIds.data(), adjacencyVals.data(), bestEdge.data(), edgeFlags.data());

	for (size_t numIterations = 0; numIterations < (oStride + 1) / 2; ++numIterations)
	{
		superNodeIds[oStride] = 0;

		thrust::counting_iterator<size_t> firstNode(0u);
		thrust::counting_iterator<size_t> lastNode(oStride);
	
		thrust::for_each(firstNode, lastNode, edgeVote);
		
		if (superNodeIds[oStride] != 0)
		{
			thrust::for_each(firstNode, lastNode, updateSuperNodes);
//#ifdef _DEBUG
//			outputDeviceVector("supernode ids: ", superNodeIds);
//#endif
		}
		else
		{
			break;
		}
	}
	
	superNodeIds.clear();

	//write output
	oAdjacencyMatrix = thrust::device_vector<EdgeType>(oStride * oStride, NOT_CONNECTED);

	SpanningTreeEdgeInitializer initEdgeTypes(oStride, oAdjacencyMatrix.data());

	thrust::for_each(
		thrust::make_zip_iterator(thrust::make_tuple(adjacencyKeys.begin(), adjacencyVals.begin(), edgeFlags.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(adjacencyKeys.end(), adjacencyVals.end(), edgeFlags.end())),
		initEdgeTypes);

	SpanningTreeEdgeWriter writeEdgeTypes(oStride, oAdjacencyMatrix.data());

	thrust::for_each(
		thrust::make_zip_iterator(thrust::make_tuple(adjacencyKeys.begin(), adjacencyVals.begin(), edgeFlags.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(adjacencyKeys.end(), adjacencyVals.end(), edgeFlags.end())),
		writeEdgeTypes);
}

__host__ void Graph::fromAdjacencyList(size_t aNumNodes)
{
	intervals = thrust::device_vector<unsigned int>(aNumNodes + 1);
	//initialize first value + all empty intervals at the start
	thrust::fill(intervals.begin(), intervals.begin() + adjacencyKeys[0] + 1, 0u);
	//initialize last value + all empty intervals at the end
	thrust::fill(intervals.end() - 1 - (aNumNodes - 1 - adjacencyKeys[adjacencyKeys.size() - 1]), intervals.end(), (unsigned int)adjacencyKeys.size());

	IntervalExtractor extractIntervals(intervals.data());
	thrust::counting_iterator<size_t> first(0u);
	thrust::counting_iterator<size_t> last(adjacencyKeys.size());

	thrust::for_each(
		thrust::make_zip_iterator(thrust::make_tuple(adjacencyKeys.begin(), adjacencyKeys.begin() + 1, first)),
		thrust::make_zip_iterator(thrust::make_tuple(adjacencyKeys.end() - 1, adjacencyKeys.end(), last)),
		extractIntervals);

//#ifdef _DEBUG
//	outputDeviceVector("Edge X: ", adjacencyKeys);
//	outputDeviceVector("Edge Y: ", adjacencyVals);
//	outputDeviceVector("Extracted intervals: ", intervals);
//#endif
}

// A recursive function that uses visited[] and parent to detect
// cycle in subgraph reachable from vertex v.
__host__ bool isCyclicUtil(
	unsigned int v,
	thrust::host_vector<unsigned int>& visited,
	unsigned int parent,
	thrust::host_vector<unsigned int>& intervalsHost,
	thrust::host_vector<unsigned int>& adjacencyValsHost,
	thrust::host_vector<Graph::EdgeType>& adjacencyMatrixType)
{
	// Mark the current node as visited
	visited[v] = 1u;

	unsigned int numNodes = (unsigned int)intervalsHost.size() - 1;
	// Recur for all the vertices adjacent to this vertex
	for (unsigned int nbrId = intervalsHost[v]; nbrId < intervalsHost[v + 1]; ++nbrId)
	{
		unsigned int nbrNodeId = adjacencyValsHost[nbrId];
		if (adjacencyMatrixType[nbrNodeId + numNodes * v] != Graph::EdgeType::SPANNING_TREE)
			continue;
		// If an adjacent is not visited, then recur for that adjacent
		if (visited[nbrNodeId] == 0)
		{
			if (isCyclicUtil(
				nbrNodeId, 
				visited,
				v,
				intervalsHost,
				adjacencyValsHost,
				adjacencyMatrixType
				))
				return true;
		}
		// If an adjacent is visited and not parent of current vertex,
		// then there is a cycle.
		else if (nbrNodeId != parent)
			return true;
	}
	return false;
}

__host__ int Graph::testGraphConstruction(int aGraphSize)
{
	thrust::host_vector<unsigned int> adjacencyMatrixHost(aGraphSize * aGraphSize);
	for (size_t i = 0; i < aGraphSize; ++i)
	{
		for (size_t j = 0; j < i; ++j)
		{
			bool makeEdge = rand() / RAND_MAX > 0.25f;
			if (makeEdge)
			{
				adjacencyMatrixHost[j * aGraphSize + i] = 1u;
				adjacencyMatrixHost[i * aGraphSize + j] = 1u;
			}
		}
	}

	cudastd::timer timer;

	thrust::device_vector<unsigned int> adjacencyMatrixDevice(aGraphSize * aGraphSize);
	thrust::copy(adjacencyMatrixHost.begin(), adjacencyMatrixHost.end(), adjacencyMatrixDevice.begin());
	//Graph testGraph;
	//testGraph.fromAdjacencyMatrix(adjacencyMatrixDevice, (size_t)aGraphSize);
	fromAdjacencyMatrix(adjacencyMatrixDevice, (size_t)aGraphSize);
	adjacencyMatrixDevice.clear();
	size_t newGrapSize;
	//testGraph.toAdjacencyMatrix(adjacencyMatrixDevice, newGrapSize);
	toAdjacencyMatrix(adjacencyMatrixDevice, newGrapSize);
	
	float totalTime = timer.get();
	
	thrust::host_vector<unsigned int> adjacencyMatrixHostCpy(adjacencyMatrixDevice.begin(), adjacencyMatrixDevice.end());
	for (size_t i = 0; i < aGraphSize * aGraphSize; ++i)
	{
		if (adjacencyMatrixHostCpy[i] != adjacencyMatrixHost[i])
		{
			std::cerr << "Wrong adjacency matrix value at position " << i;
			std::cerr << " device " << adjacencyMatrixDevice[i] << " ";
			std::cerr << " host " << adjacencyMatrixHost[i] << " ";
			std::cerr << "\n";
			return 1;
		}
	}

	std::cerr << "Converted graph to and from adjacency matrix in "<< totalTime << "ms\n";

	return testSpanningTreeConstruction();
}

__host__ int Graph::testSpanningTreeConstruction()
{
	cudastd::timer timer;
	
	size_t graphSize;
	thrust::device_vector<Graph::EdgeType> adjacencyMatrixType;

	toSpanningTree(adjacencyMatrixType, graphSize);

	float totalTreeTime = timer.get();
	timer.cleanup();

	thrust::host_vector<Graph::EdgeType> adjacencyMatrixTypeHost(adjacencyMatrixType.begin(), adjacencyMatrixType.end());

	//test for cycles

	// Mark all the vertices as not visited and not part of recursion
	// stack
	thrust::host_vector<unsigned int> visited(graphSize, 0);
	thrust::host_vector<unsigned int> intervalsHost(intervals.begin(), intervals.end());
	thrust::host_vector<unsigned int> adjacencyValsHost(adjacencyVals.begin(), adjacencyVals.end());

	// Call the recursive helper function to detect cycle in different
	// DFS trees
	for (int u = 0; u < graphSize; u++)
	{
		if (!visited[u]) // Don't recur for u if it is already visited
			if (isCyclicUtil(u, visited, (unsigned int)-1, intervalsHost, adjacencyValsHost, adjacencyMatrixTypeHost))
			{
				std::cerr << "Wrong spanning tree - has cycle containing node " << u << "\n";
				return 3;
			}
	}

	for (size_t nodeId = 0; nodeId < graphSize; ++nodeId)
	{
		std::vector<size_t> DFSFrontierD1;
		for (size_t neighborId = 0; neighborId < graphSize; ++neighborId)
		{
			if (neighborId == nodeId)
				continue;
			if (adjacencyMatrixTypeHost[neighborId + graphSize * nodeId] == Graph::EdgeType::SPANNING_TREE)
				DFSFrontierD1.push_back(neighborId);
		}

		if (DFSFrontierD1.empty())
		{
			for (size_t neighborId = 0; neighborId < graphSize; ++neighborId)
			{
				if (neighborId == nodeId)
					continue;
				if (adjacencyMatrixTypeHost[neighborId + graphSize * nodeId] == Graph::EdgeType::CYCLE)
				{
					std::cerr << "Wrong spanning tree - contains isolated node " << nodeId << "\n";
					return 4;
				}
			}
		}
	}

	std::cerr << "Computed graph spanning tree in " << totalTreeTime << "ms\n";

	return 0;	
}


