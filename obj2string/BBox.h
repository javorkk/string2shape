#ifdef _MSC_VER
#pragma once
#endif

#ifndef BBOX_H_B1A28A7D_7D2D_47CF_B970_1C70D54C6DEB
#define BBOX_H_B1A28A7D_7D2D_47CF_B970_1C70D54C6DEB


#include <cfloat> //FLT_MAX
#include "Algebra.h"
#include "Primitive.h"

//An axis aligned bounding box
class BBox
{
public:
	float3 vtx[2]; //inherited


	//Returns the entry and exit distances of the ray with the
	//	bounding box.
	//If the first returned distance > the second, than
	//	the ray does not intersect the bounding box at all
	__device__ __host__ void clip(const float3 &aRayOrg, const float3& aRayDir, float& oEntry, float& oExit) const
	{
		const float3 t1 = (vtx[0] - aRayOrg) / aRayDir;
		float3 tMax = (vtx[1] - aRayOrg) / aRayDir;

		const float3 tMin = min(t1, tMax);
		tMax = max(t1, tMax);

		oEntry = fmaxf(fmaxf(tMin.x, tMin.y), tMin.z);
		oExit = fminf(fminf(tMax.x, tMax.y), tMax.z);

	}

	__device__ __host__ void fastClip(const float3 &aRayOrg, const float3& aRayDirRCP, float& oEntry, float& oExit) const
	{
		const float3 t1 = (vtx[0] - aRayOrg) * aRayDirRCP;
		float3 tMax = (vtx[1] - aRayOrg) * aRayDirRCP;

		const float3 tMin = min(t1, tMax);
		tMax = max(t1, tMax);

		oEntry = fmaxf(fmaxf(tMin.x, tMin.y), tMin.z);
		oExit = fminf(fminf(tMax.x, tMax.y), tMax.z);

	}

	//Extend the bounding box with a point
	__device__ __host__ void extend(const float3 &aPoint)
	{
		vtx[0] = min(vtx[0], aPoint);
		vtx[1] = max(vtx[1], aPoint);
	}

	//Extend the bounding box with another bounding box
	__device__ __host__ void extend(const BBox &aBBox)
	{
		vtx[0] = min(vtx[0], aBBox.vtx[0]);
		vtx[1] = max(vtx[1], aBBox.vtx[1]);
	}

	//Tighten the bounding box around another bounding box
	__device__ __host__ void tighten(const BBox &aBBox)
	{
		vtx[0] = max(vtx[0], aBBox.vtx[0]);
		vtx[1] = min(vtx[1], aBBox.vtx[1]);
	}

	//Tighten the bounding box around two points
	__device__ __host__ void tighten(const float3 &aMin, const float3 &aMax)
	{
		vtx[0] = max(vtx[0], aMin);
		vtx[1] = min(vtx[1], aMax);
	}


	//Returns an "empty" bounding box. Extending such a bounding
	//	box with a point will always create a bbox around the point
	//	and with a bbox - will simply copy the bbox.
	__device__ __host__ static BBox empty()
	{
		BBox ret;
		ret.vtx[0].x = FLT_MAX;
		ret.vtx[0].y = FLT_MAX;
		ret.vtx[0].z = FLT_MAX;
		ret.vtx[1].x = -FLT_MAX;
		ret.vtx[1].y = -FLT_MAX;
		ret.vtx[1].z = -FLT_MAX;
		return ret;
	}

	__device__ __host__ const float3 diagonal() const
	{
		return vtx[1] - vtx[0];
	}
};

template<class tPrimitive>
class BBoxExtractor
{
public:
	__device__ __host__ static BBox get(const tPrimitive& aPrimitive)
	{
		BBox result = BBox::empty();

		//#pragma unroll 3
		for (uint i = 0; i < tPrimitive::NUM_VERTICES; ++i)
		{
			result.extend(aPrimitive.vtx[i]);
		}

		return result;
	}
};

#endif // BBOX_H_B1A28A7D_7D2D_47CF_B970_1C70D54C6DEB
