#include "pch.h"
#include "CollisionDetector.h"

//#include "WFObject.h" //from CollisionDetector.h
//#include "Graph.h" //from CollisionDetector.h

#include "UniformGrid.h"
#include "UniformGridSortBuilder.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

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
		for (unsigned int refId = aCellRange.x; refId < aCellRange.y; ++refId)
		{
			unsigned int myPrimId = primIds[refId];
			unsigned int myObjId = objIds[myPrimId];
			for (unsigned int otherRefId = refId + 1; otherRefId < aCellRange.y; ++otherRefId)
			{
				unsigned int otherPrimId = primIds[otherRefId];
				unsigned int otherObjId = objIds[otherPrimId];
				if (myObjId != otherObjId)
					numCollisions += 2u;
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

		for (unsigned int refId = aCellRange.x; refId < aCellRange.y; ++refId)
		{
			unsigned int myPrimId = primIds[refId];
			unsigned int myObjId = objIds[myPrimId];
			for (unsigned int otherRefId = refId + 1; otherRefId < aCellRange.y; ++otherRefId)
			{
				unsigned int otherPrimId = primIds[otherRefId];
				unsigned int otherObjId = objIds[otherPrimId];
				if (myObjId != otherObjId)
				{
					keys[outputPosition] = myObjId;
					vals[outputPosition] = otherObjId;
					++outputPosition;
					keys[outputPosition] = otherObjId;
					vals[outputPosition] = myObjId;
					++outputPosition;
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


Graph CollisionDetector::computeCollisionGraph(WFObject & aObj, float aRelativeThreshold) const
{
	//compute scene diagonal
	float3 minBound = rep( FLT_MAX);
	float3 maxBound = rep(-FLT_MAX);
	for (auto it = aObj.vertices.begin(); it != aObj.vertices.end(); ++it)
	{
		minBound = min(*it, minBound);
		maxBound = max(*it, maxBound);
	}

	float cellDiagonal = len(maxBound - minBound) * aRelativeThreshold;
	float3 res = (maxBound - minBound) / (cellDiagonal * 0.3333333f);

	//compute vertex index buffer for the triangles
	std::vector<uint3> host_indices(aObj.faces.size());
	for (size_t i = 0; i < aObj.faces.size(); i++)
	{
		host_indices[i].x = (unsigned int)aObj.faces[i].vert1;
		host_indices[i].y = (unsigned int)aObj.faces[i].vert2;
		host_indices[i].z = (unsigned int)aObj.faces[i].vert3;
	}
	//copy the vertex index buffer to the device
	thrust::device_vector<uint3> device_indices(host_indices.begin(), host_indices.end());

	UniformGridSortBuilder builder;
	UniformGrid grid = builder.build(aObj, (int)res.x, (int)res.y, (int)res.z);

	//compute per-face object id
	std::vector<unsigned int> objectIdPerFaceHost(host_indices.size());
	for (size_t i = 0; i < aObj.objects.size(); ++i)
	{
		int2 range = aObj.objects[i];
		for (size_t faceId = (size_t)range.x; faceId < (size_t)range.y; ++faceId)
		{
			objectIdPerFaceHost[faceId] = (unsigned int)i;//set object id
		}
	}
	//copy the obj ids to the device
	thrust::device_vector<unsigned int> objectIdPerFaceDevice(objectIdPerFaceHost);

	//delete all grid cells that contain primitives from a single object
	thrust::device_vector<uint2> trimmed_cells(grid.cells.size());
	CellTrimmer trimmCells(objectIdPerFaceDevice.data(), grid.primitives.data());
	thrust::transform(grid.cells.begin(), grid.cells.end(), trimmed_cells.begin(), trimmCells);

	//count all obj-obj collisions
	thrust::device_vector<unsigned int> collision_counts(grid.cells.size() + 1);
	CollisionCounter countCollisions(objectIdPerFaceDevice.data(), grid.primitives.data());
	thrust::transform(trimmed_cells.begin(), trimmed_cells.end(), collision_counts.begin(), countCollisions);

	thrust::exclusive_scan(collision_counts.begin(), collision_counts.end(), collision_counts.begin());

	//allocate storage for obj-obj collisions
	unsigned int numCollisions = collision_counts[collision_counts.size() - 1];
	thrust::device_vector<unsigned int> collision_keys(numCollisions);
	thrust::device_vector<unsigned int> collision_vals(numCollisions);

	//write all obj-obj collisions
	CollisionWriter writeCollisions(objectIdPerFaceDevice.data(), grid.primitives.data(),
		collision_keys.data(), collision_vals.data());
	
	thrust::for_each(
		thrust::make_zip_iterator(thrust::make_tuple(trimmed_cells.begin(), collision_counts.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(trimmed_cells.end(), collision_counts.end() - 1)),
		writeCollisions);

	//sort all obj-obj collisions
	//sort the pairs
	thrust::sort_by_key(collision_keys.begin(), collision_keys.end(), collision_vals.begin());

	//remove all duplicate obj-obj collisions
	//thrust::device_vector<unsigned int> collision_keys_unique;
	//thrust::device_vector<unsigned int> collision_vals_unique;
	auto begin_iterator = thrust::make_zip_iterator(thrust::make_tuple(collision_keys.begin(), collision_vals.begin()));
	auto end_iterator = thrust::unique_copy(
		begin_iterator,
		thrust::make_zip_iterator(thrust::make_tuple(collision_keys.end(), collision_vals.end())),
		begin_iterator,
		isEqualCollision());

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
	
	return result;
}
