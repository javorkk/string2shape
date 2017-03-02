#ifdef _MSC_VER
#pragma once
#endif

#ifndef UNIFORMGRID_H_5A89108F_6F07_4F96_81EB_63A65F538434
#define UNIFORMGRID_H_5A89108F_6F07_4F96_81EB_63A65F538434

#include <thrust/device_vector.h>
#include "Algebra.h"

class UniformGrid
{
public:
	float3 vtx[2]; //inherited -> bounding box
	int res[3];
	//float3 cellSize;
	//float3 cellSizeRCP;
	thrust::device_vector<uint2> cells;
	thrust::device_vector<unsigned int> primitives;

	//uint  numPrimitiveReferences;

	__host__ __device__ const float3 getResolution() const
	{
		float3 retval;
		retval.x = static_cast<float>(res[0]);
		retval.y = static_cast<float>(res[1]);
		retval.z = static_cast<float>(res[2]);
		return retval;
	}


	__host__ __device__ float3 getCellSize() const
	{
		//return cellSize;
		return fastDivide(vtx[1] - vtx[0], getResolution());
	}

	__host__ __device__ void setCellSize(const float3& aCellSize)
	{
		//set the variable if it exists
		//cellSize = aCellSize;
		//...do nothing
	}

	__host__ __device__ float3 getCellSizeRCP() const
	{
		//return cellSizeRCP;
		return fastDivide(getResolution(), vtx[1] - vtx[0]);
	}

	__host__ __device__ void setCellSizeRCP(const float3& aCellSizeRCP)
	{
		//set the variable if it exits
		//cellSizeRCP = aCellSizeRCP;
		//...or do nothing
	}

	//convert a 3D cell index into a linear one
	__host__ __device__ int getCellIdLinear(int aIdX, int aIdY, int aIdZ) const
	{
		return aIdX + aIdY * res[0] + aIdZ * res[0] * res[1];
	}

	//convert a 3D cell index into a linear one
	__host__ __device__ int3 getCellId3D(int aLinearId) const
	{
		return make_int3(
			aLinearId % res[0],
			(aLinearId % (res[0] * res[1])) / res[0],
			aLinearId / (res[0] * res[1]));
	}

	__host__ __device__ uint2 getCell(int aIdX, int aIdY, int aIdZ) const
	{
		return cells[aIdX + aIdY * res[0] + aIdZ * res[0] * res[1]];
	}

	__host__ __device__ void setCell(int aIdX, int aIdY, int aIdZ, uint2 aVal)
	{
		cells[aIdX + aIdY * res[0] + aIdZ * res[0] * res[1]] = aVal;
	}

	__host__ __device__ int3 getCellIdAt(float3 aPosition) const
	{
		float3 cellIdf = (aPosition - vtx[0]) * getCellSizeRCP();
		int3 cellId;
		cellId.x = static_cast<int>(cellIdf.x);
		cellId.y = static_cast<int>(cellIdf.y);
		cellId.z = static_cast<int>(cellIdf.z);
		return cellId;
	}

	__host__ __device__ uint2 getCellAt(float3 aPosition) const
	{
		float3 cellIdf = (aPosition - vtx[0]) * getCellSizeRCP();
		return getCell(static_cast<int>(cellIdf.x), static_cast<int>(cellIdf.y), static_cast<int>(cellIdf.z));
	}

	__host__ __device__ float3 getCellCenter(int aIdX, int aIdY, int aIdZ) const
	{
		float3 cellIdf = make_float3((float)aIdX + 0.5f, (float)aIdY + 0.5f, (float)aIdZ + 0.5f);
		return vtx[0] + cellIdf * getCellSize();
	}

	__host__ __device__ unsigned int getPrimitiveId(unsigned int aId) const
	{
		return primitives[aId];
	}
};

#endif // UNIFORMGRID_H_5A89108F_6F07_4F96_81EB_63A65F538434
