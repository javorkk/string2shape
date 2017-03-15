#include "pch.h"
#include "Graph.h"

#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/scan.h>
#include <thrust/transform_scan.h>
#include <thrust/fill.h>

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

class EdgeTypeMatrixWriter
{
public:
	size_t stride;
	thrust::device_ptr<Graph::EdgeType> matrix;
	thrust::device_ptr<unsigned int> adjIntervals;
	thrust::device_ptr<unsigned int> neighborIds;


	EdgeTypeMatrixWriter(
		size_t aStride,
		thrust::device_ptr<Graph::EdgeType> aMatrix,
		thrust::device_ptr<unsigned int> aIntervals,
		thrust::device_ptr<unsigned int> aNeighbors
		) :stride(aStride), matrix(aMatrix),
		adjIntervals(aIntervals), neighborIds(aNeighbors)
	{}

	template <typename Tuple>
	__host__ __device__	void operator()(Tuple t)
	{
		const unsigned int myNodeId0 = thrust::get<0>(t);
		const unsigned int myNodeId1 = thrust::get<1>(t);

		//make sure node 0 has neighbors
		if (adjIntervals[myNodeId0] >= adjIntervals[myNodeId0 + 1])
		{
			matrix[myNodeId0 + myNodeId1 * stride] = Graph::EdgeType::NOT_CONNECTED;
			return;//isolated node1
		}
		//make sure node 1 has neighbors
		if (adjIntervals[myNodeId1] >= adjIntervals[myNodeId1 + 1])
		{
			matrix[myNodeId0 + myNodeId1 * stride] = Graph::EdgeType::NOT_CONNECTED;
			return;//isolated node2
		}
		//the spanning tree connects each node to its first neighbor in the adjacency list
		const unsigned int bestNeighbor0 = neighborIds[adjIntervals[myNodeId0]];
		const unsigned int bestNeighbor1 = neighborIds[adjIntervals[myNodeId1]];
		bool edgeOnTree = bestNeighbor0 == myNodeId1 || bestNeighbor1 == myNodeId0;

		matrix[myNodeId0 + myNodeId1 * stride] = edgeOnTree ? Graph::EdgeType::SPANNING_TREE : Graph::EdgeType::CYCLE;

		//if (bestNeighbor0 == myNodeId1 || bestNeighbor1 == myNodeId0)
		//{
		//	matrix[myNodeId0 + myNodeId1 * stride] = Graph::EdgeType::SPANNING_TREE;
		//	return;
		//}

		////nodes are not on the spanning tree, if connected mark the edge as cycle
		//bool connected = false;
		//for (unsigned int nbrId = adjIntervals[myNodeId0] + 1; nbrId < adjIntervals[myNodeId0 + 1] && !connected; ++nbrId)
		//{
		//	if (neighborIds[nbrId] == myNodeId1)
		//		connected = true;
		//}
		//
		//matrix[myNodeId0 + myNodeId1 * stride] = connected ? Graph::EdgeType::CYCLE : Graph::EdgeType::NOT_CONNECTED;
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
	unsigned int stride;
	thrust::device_ptr<unsigned int> matPrefix;
	thrust::device_ptr<unsigned int> intervals;
public:
	MatrixRowCopy(
		unsigned int aStride,
		thrust::device_ptr<unsigned int> aPrefix,
		thrust::device_ptr<unsigned int> aIntervals
	) : stride(aStride), matPrefix(aPrefix), intervals(aIntervals)
	{}

	__host__ __device__	void operator()(unsigned int aRowId)
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
	thrust::counting_iterator<size_t> lastRowIdP1(aStride + 1);
	MatrixRowCopy copyOp(aStride, matrixPrefix.data(), intervals.data());
	thrust::for_each(firstRowId, lastRowIdP1, copyOp);

	//for (size_t rowId = 0; rowId < aStride + 1; ++rowId)
	//{
	//	intervals[rowId] = matrixPrefix[aStride * rowId];
	//}


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

__host__ void Graph::toTypedAdjacencyMatrix(thrust::device_vector<EdgeType>& oAdjacencyMatrix, size_t & oStride)
{
	if (intervals.size() < 2u)
		return; //emtpy graph

	oStride = intervals.size() - 1u;
	oAdjacencyMatrix = thrust::device_vector<EdgeType>(oStride * oStride, NOT_CONNECTED);
	
	EdgeTypeMatrixWriter writeEdgeTypes(oStride, oAdjacencyMatrix.data(), intervals.data(), adjacencyVals.data());

	thrust::for_each(
		thrust::make_zip_iterator(thrust::make_tuple(adjacencyKeys.begin(), adjacencyVals.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(adjacencyKeys.end(), adjacencyVals.end())),
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

#ifdef _DEBUG
	outputDeviceVector("Edge X: ", adjacencyKeys);
	outputDeviceVector("Edge Y: ", adjacencyVals);

	outputDeviceVector("Extracted intervals: ", intervals);
#endif
}

__host__ int Graph::testGraphConstruction(int aGraphSize)
{
	thrust::host_vector<unsigned int> adjacencyMatrixHost(aGraphSize * aGraphSize);
	for (size_t i = 0; i < aGraphSize; ++i)
	{
		for (size_t j = 0; j < i; ++j)
		{
			bool makeEdge = rand() / RAND_MAX > 0.5f;
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
	timer.cleanup();
	
	for (size_t i = 0; i < aGraphSize * aGraphSize; ++i)
	{
		if (adjacencyMatrixDevice[i] != adjacencyMatrixHost[i])
		{
			std::cerr << "Wrong adjacency matrix value at position " << i;
			std::cerr << " device " << adjacencyMatrixDevice[i] << " ";
			std::cerr << " host " << adjacencyMatrixHost[i] << " ";
			std::cerr << "\n";
			return 1;
		}
	}

	std::cerr << "Converted graph to and from adjacency matrix in "<< totalTime << "ms\n";

	return 0;
}
