#include "pch.h"
#include "CollisionDetector.h"

//#include "WFObject.h" //from CollisionDetector.h
//#include "Graph.h" //from CollisionDetector.h
#include "WFObjUtils.h"
#include "UniformGrid.h"
#include "UniformGridSortBuilder.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "DebugUtils.h"
#include "Timer.h"
#include <thrust/functional.h>


UniformGridSortBuilder builder;

class nonEmptyCell
{
public:

	template <typename Tuple>
	__host__ __device__	bool operator()(Tuple t1)
	{
		const unsigned int range_start = thrust::get<0>(t1);
		const unsigned int range_end = thrust::get<1>(t1);
		return range_start < range_end;
	}

};

class nonEmptyRange
{
public:

	__host__ __device__	bool operator()(const uint2& aCellRange)
	{
		const unsigned int range_start = aCellRange.x;
		const unsigned int range_end = aCellRange.y;
		return range_start < range_end;
	}
};

class CellTrimmer
{
public:
	thrust::device_ptr<unsigned int>  objIds;
	thrust::device_ptr<unsigned int>  primIds;

	CellTrimmer(
		thrust::device_ptr<unsigned int> aObjIds,
		thrust::device_ptr<unsigned int> aPrimIds
	) :
		objIds(aObjIds),
		primIds(aPrimIds)
	{}

	__host__ __device__	uint2 operator()(const uint2& aCellRange)
	{
		if (aCellRange.x + 1 >= aCellRange.y)
			return make_uint2(0u, 0u); //one or two primitives inside => no collision

		bool hasCollisionCandidates = false;

		for (unsigned int refId = aCellRange.x; refId < aCellRange.y - 1 && !hasCollisionCandidates; ++refId)
		{
			unsigned int primId = primIds[refId];
			unsigned int objId = objIds[primId];
			unsigned int nextPrimId = primIds[refId + 1];
			unsigned int nextObjId = objIds[nextPrimId];
			if (objId != nextObjId)
				hasCollisionCandidates = true;		
		}
		if (hasCollisionCandidates)
			return aCellRange;
		else
			return make_uint2(0u,0u);
	}

};

class CollisionOperator
{
public:
	thrust::device_ptr<unsigned int>  objIds;
	thrust::device_ptr<unsigned int>  primIds;

	unsigned int stride;
	thrust::device_ptr<unsigned int>  adjMatrix;

	CollisionOperator(
		thrust::device_ptr<unsigned int> aObjIds,
		thrust::device_ptr<unsigned int> aPrimIds,
		unsigned int					 aStride,
		thrust::device_ptr<unsigned int> aMatrix
	) :
		objIds(aObjIds),
		primIds(aPrimIds),
		stride(aStride),
		adjMatrix(aMatrix)
	{}

	__host__ __device__	void operator()(const uint2& aCellRange)
	{
		if (aCellRange.x >= aCellRange.y)
			return;

		uint2 lastRecordedPair = make_uint2(objIds[primIds[aCellRange.x]], objIds[primIds[aCellRange.x]]);
		for (unsigned int refId = aCellRange.x; refId < aCellRange.y; ++refId)
		{
			unsigned int myPrimId = primIds[refId];
			unsigned int myObjId = objIds[myPrimId];
			for (unsigned int otherRefId = refId + 1; otherRefId < aCellRange.y; ++otherRefId)
			{
				unsigned int otherPrimId = primIds[otherRefId];
				unsigned int otherObjId = objIds[primIds[otherRefId]];
				if (myObjId != otherObjId &&
					!(myObjId == lastRecordedPair.x && otherObjId == lastRecordedPair.y ||
						myObjId == lastRecordedPair.y && otherObjId == lastRecordedPair.x)
					)
				{
					adjMatrix[myObjId + stride * otherObjId] = 1u;
					adjMatrix[otherObjId + stride * myObjId] = 1u;
					lastRecordedPair.x = myObjId;
					lastRecordedPair.y = otherObjId;
				}
			}
		}
	}

};

class CollisionCounter
{
public:
	thrust::device_ptr<unsigned int>  objIds;
	thrust::device_ptr<unsigned int>  primIds;

	CollisionCounter(
		thrust::device_ptr<unsigned int> aObjIds,
		thrust::device_ptr<unsigned int> aPrimIds
		) :
		objIds(aObjIds),
		primIds(aPrimIds)
	{}

	__host__ __device__	unsigned int operator()(const uint2& aCellRange)
	{
		if (aCellRange.x >= aCellRange.y)
			return 0u;

		unsigned int numCollisions = 0u;
		uint2 lastRecordedPair = make_uint2(objIds[primIds[aCellRange.x]], objIds[primIds[aCellRange.x]]);

		for (unsigned int refId = aCellRange.x; refId < aCellRange.y; ++refId)
		{
			unsigned int myObjId = objIds[primIds[refId]];
			for (unsigned int otherRefId = refId + 1; otherRefId < aCellRange.y; ++otherRefId)
			{
				unsigned int otherObjId = objIds[primIds[otherRefId]];
				if (myObjId != otherObjId &&
					!(myObjId == lastRecordedPair.x && otherObjId == lastRecordedPair.y ||
						myObjId == lastRecordedPair.y && otherObjId == lastRecordedPair.x)
					)
				{
					numCollisions += 2u;
					lastRecordedPair.x = myObjId;
					lastRecordedPair.y = otherObjId;
				}
			}
		}
		return numCollisions;
	}

};

class CollisionWriter
{
public:

	thrust::device_ptr<unsigned int>  objIds;
	thrust::device_ptr<unsigned int>  primIds;

	thrust::device_ptr<unsigned int>  keys;
	thrust::device_ptr<unsigned int>  vals;


	CollisionWriter(
		thrust::device_ptr<unsigned int> aObjIds,
		thrust::device_ptr<unsigned int> aPrimIds,
		thrust::device_ptr<unsigned int> aKeys,
		thrust::device_ptr<unsigned int> aVals
	) :
		objIds(aObjIds),
		primIds(aPrimIds),
		keys(aKeys),
		vals(aVals)
	{}


	template <typename Tuple>
	__host__ __device__	void operator()(Tuple t)
	{
		const uint2 aCellRange = thrust::get<0>(t);
		unsigned int outputPosition = thrust::get<1>(t);
		
		if (aCellRange.x >= aCellRange.y)
			return;

		uint2 lastRecordedPair = make_uint2(objIds[primIds[aCellRange.x]], objIds[primIds[aCellRange.x]]);
		for (unsigned int refId = aCellRange.x; refId < aCellRange.y; ++refId)
		{
			unsigned int myObjId = objIds[primIds[refId]];
			for (unsigned int otherRefId = refId + 1; otherRefId < aCellRange.y; ++otherRefId)
			{

				unsigned int otherObjId = objIds[primIds[otherRefId]];
				if (myObjId != otherObjId && 
					!(myObjId == lastRecordedPair.x && otherObjId == lastRecordedPair.y ||
					myObjId == lastRecordedPair.y && otherObjId == lastRecordedPair.x)
					)
				{
					keys[outputPosition] = myObjId;
					vals[outputPosition] = otherObjId;
					++outputPosition;
					keys[outputPosition] = otherObjId;
					vals[outputPosition] = myObjId;
					++outputPosition;
					lastRecordedPair.x = myObjId;
					lastRecordedPair.y = otherObjId;
				}
			}
		}
	}

};

class isEqualCollision
{
public:

	template <typename Tuple>
	__host__ __device__	bool operator()(Tuple t1, Tuple t2)
	{
		const unsigned int key1 = thrust::get<0>(t1);
		const unsigned int val1 = thrust::get<1>(t1);

		const unsigned int key2 = thrust::get<0>(t2);
		const unsigned int val2 = thrust::get<1>(t2);

		return key1 == key2 && val1 == val2;
	}

};


Graph CollisionDetector::computeCollisionGraph(WFObject & aObj, float aRelativeThreshold)
{
	cudastd::timer timer;
	cudastd::timer intermTimer;

	//compute scene diagonal
	float3 minBound, maxBound;
	ObjectBoundsExporter()(aObj, minBound, maxBound);
	float boundsDiagonal = len(maxBound - minBound);
	float3 res = (maxBound - minBound) / (boundsDiagonal * 0.577350269f * aRelativeThreshold); //0.577350269 ~ sqrtf(3.f)

	UniformGrid grid = builder.build(aObj, (int)res.x, (int)res.y, (int)res.z);
	
//#ifdef _DEBUG
//	builder.test(grid, aObj);
//#endif

	//compute per-face object id
	thrust::host_vector<unsigned int> objectIdPerFaceHost(aObj.faces.size());
	for (size_t i = 0; i < aObj.objects.size(); ++i)
	{
		int2 range = aObj.objects[i];
		for (int faceId = range.x; faceId < range.y; ++faceId)
		{
			objectIdPerFaceHost[faceId] = (unsigned int)i;//set object id
		}
	}

	//copy the obj ids to the device
	thrust::device_vector<unsigned int> objectIdPerFaceDevice(aObj.faces.size());
	thrust::copy(objectIdPerFaceHost.begin(), objectIdPerFaceHost.end(), objectIdPerFaceDevice.begin());

	initTime = intermTimer.get();
	intermTimer.start();

//#ifdef _DEBUG
//	outputDeviceVector("Obj id per face: ", objectIdPerFaceDevice);
//#endif

	//delete all grid cells that contain primitives from a single object
	thrust::device_vector<uint2> trimmed_cells(grid.cells.size());
	CellTrimmer trimmCells(objectIdPerFaceDevice.data(), grid.primitives.data());
	thrust::transform(grid.cells.begin(), grid.cells.end(), trimmed_cells.begin(), trimmCells);

	auto trimmed_cells_end = thrust::copy_if(trimmed_cells.begin(), trimmed_cells.end(), trimmed_cells.begin(), nonEmptyRange());

	trimmTime = intermTimer.get();
	intermTimer.start();

//#ifdef _DEBUG
//	thrust::device_vector<unsigned int> trimmed_cells_x(grid.cells.size());
//	thrust::device_vector<unsigned int> trimmed_cells_y(grid.cells.size());
//	thrust::transform(trimmed_cells.begin(), trimmed_cells.end(), trimmed_cells_x.begin(), uint2_get_x());
//	thrust::transform(trimmed_cells.begin(), trimmed_cells.end(), trimmed_cells_y.begin(), uint2_get_y());
//	auto begin_iterator_dbg = thrust::make_zip_iterator(thrust::make_tuple(trimmed_cells_x.begin(), trimmed_cells_y.begin()));
//	auto end_iterator_dbg = thrust::copy_if(
//		begin_iterator_dbg,
//		thrust::make_zip_iterator(thrust::make_tuple(trimmed_cells_x.end(), trimmed_cells_y.end())),
//		begin_iterator_dbg,
//		nonEmptyCell());
//	thrust::device_vector<unsigned int> non_empty_cells_x(end_iterator_dbg - begin_iterator_dbg);
//	thrust::device_vector<unsigned int> non_empty_cells_y(end_iterator_dbg - begin_iterator_dbg);
//	thrust::copy(
//		begin_iterator_dbg,
//		end_iterator_dbg,
//		thrust::make_zip_iterator(thrust::make_tuple(non_empty_cells_x.begin(), non_empty_cells_y.begin()))
//	);
//	outputDeviceVector("Non-empty cells x: ", non_empty_cells_x);
//	outputDeviceVector("Non-empty cells y: ", non_empty_cells_y);
//#endif // _DEBUG

#define SINGLE_KERNEL_COLLISION
#ifdef  SINGLE_KERNEL_COLLISION //faster than multi-kernel approach
	thrust::device_vector<unsigned int> adjMatrix(aObj.objects.size() * aObj.objects.size());
	CollisionOperator collide(
		objectIdPerFaceDevice.data(),
		grid.primitives.data(),
		(unsigned int)aObj.objects.size(),
		adjMatrix.data());

	thrust::for_each(grid.cells.begin(), grid.cells.end(), collide);

	adjMatTime = intermTimer.get();
	intermTimer.start();

	//build a collision graph
	Graph result;

	result.fromAdjacencyMatrix(adjMatrix, aObj.objects.size());
#else
	//count all obj-obj collisions
	thrust::device_vector<unsigned int> collision_counts(grid.cells.size() + 1);
	CollisionCounter countCollisions(objectIdPerFaceDevice.data(), grid.primitives.data());
	thrust::transform(trimmed_cells.begin(), trimmed_cells.end(), collision_counts.begin(), countCollisions);

//#ifdef _DEBUG
//	outputDeviceVector("Collision counts: ", collision_counts);
//#endif

	thrust::exclusive_scan(collision_counts.begin(), collision_counts.end(), collision_counts.begin());

//#ifdef _DEBUG
//	outputDeviceVector("Scanned counts  : ", collision_counts);
//#endif

	//allocate storage for obj-obj collisions
	unsigned int numCollisions = collision_counts[collision_counts.size() - 1];

	countTime = intermTimer.get();
	intermTimer.start();

	thrust::device_vector<unsigned int> collision_keys(numCollisions);
	thrust::device_vector<unsigned int> collision_vals(numCollisions);

	//write all obj-obj collisions
	CollisionWriter writeCollisions(objectIdPerFaceDevice.data(), grid.primitives.data(),
		collision_keys.data(), collision_vals.data());

	thrust::for_each(
		thrust::make_zip_iterator(thrust::make_tuple(trimmed_cells.begin(), collision_counts.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(trimmed_cells.end(), collision_counts.end() - 1)),
		writeCollisions);

#ifdef _DEBUG
	outputDeviceVector("Collision keys: ", collision_keys);
	outputDeviceVector("Collision vals: ", collision_vals);
#endif

	writeTime = intermTimer.get();
	intermTimer.start();


	//sort all obj-obj collisions
	//sort the pairs
	thrust::sort_by_key(collision_vals.begin(), collision_vals.end(), collision_keys.begin());
	thrust::stable_sort_by_key(collision_keys.begin(), collision_keys.end(), collision_vals.begin());

#ifdef _DEBUG
	outputDeviceVector("Sorted keys: ", collision_keys);
	outputDeviceVector("Sorted vals: ", collision_vals);
#endif

	sortTime = intermTimer.get();
	intermTimer.start();

	//remove all duplicate obj-obj collisions
	//thrust::device_vector<unsigned int> collision_keys_unique;
	//thrust::device_vector<unsigned int> collision_vals_unique;
	auto begin_iterator = thrust::make_zip_iterator(thrust::make_tuple(collision_keys.begin(), collision_vals.begin()));
	auto end_iterator = thrust::unique_copy(
		begin_iterator,
		thrust::make_zip_iterator(thrust::make_tuple(collision_keys.end(), collision_vals.end())),
		begin_iterator,
		isEqualCollision());

	uniqueTime = intermTimer.get();
	intermTimer.start();

	//build a collision graph
	Graph result;
	
	result.adjacencyKeys = thrust::device_vector<unsigned int>(end_iterator - begin_iterator);
	result.adjacencyVals = thrust::device_vector<unsigned int>(end_iterator - begin_iterator);

	thrust::copy(
		begin_iterator,
		end_iterator,
		thrust::make_zip_iterator(thrust::make_tuple(result.adjacencyKeys.begin(), result.adjacencyVals.begin()))
	);

	result.fromAdjacencyList(aObj.objects.size());

#endif

	graphTime = intermTimer.get();
	intermTimer.cleanup();

	totalTime = timer.get();
	timer.cleanup();

	

	return result;
}

__host__ void CollisionDetector::stats()
{
	std::cerr << "Collision detection in " <<  totalTime << "ms\n";
	std::cerr << "Initialization in      " <<   initTime << "ms ";
	builder.stats();
	
	std::cerr << "Empty cells removal in " <<  trimmTime << "ms\n";
#ifdef SINGLE_KERNEL_COLLISION
	std::cerr << "Adjacency matrix in    " << adjMatTime << "ms\n";
#else
	std::cerr << "Collisions count in    " <<  countTime << "ms\n";
	std::cerr << "Collisions write in    " <<  writeTime << "ms\n";
	std::cerr << "Two-way sort in        " <<   sortTime << "ms\n";
	std::cerr << "Duplicate removal in   " << uniqueTime << "ms\n";
#endif
	std::cerr << "Graph extraction in    " <<  graphTime << "ms\n";

}

