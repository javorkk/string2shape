#include "pch.h"
#include "Graph.h"

#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/scan.h>
#include <thrust/transform_scan.h>
#include <thrust/fill.h>

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


__host__ void Graph::fromAdjacencyMatrix(thrust::device_vector<unsigned int>& aAdjacencyMatrix, size_t aStride)
{
	if (aAdjacencyMatrix.size() != aStride * aStride)
		return; //something is wrong


	thrust::device_vector<unsigned int> matrixPrefix(aAdjacencyMatrix.size() + 1);
	thrust::transform(aAdjacencyMatrix.begin(), aAdjacencyMatrix.end(), matrixPrefix.begin(), isEdge());
	thrust::exclusive_scan(matrixPrefix.begin(), matrixPrefix.end(), matrixPrefix.begin());

	intervals = thrust::device_vector<unsigned int>(aStride + 1);
	for (size_t rowId = 0; rowId < aStride + 1; ++rowId)
	{
		intervals[rowId] = matrixPrefix[aStride * rowId];
	}

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
		thrust::make_zip_iterator(thrust::make_tuple(adjacencyKeys.begin(), adjacencyVals.begin() + 1, first)),
		thrust::make_zip_iterator(thrust::make_tuple(adjacencyKeys.end() - 1, adjacencyVals.end(), last)),
		extractIntervals);
}
