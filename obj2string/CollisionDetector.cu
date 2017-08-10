#include "pch.h"
#include "CollisionDetector.h"

//#include "WFObject.h" //from CollisionDetector.h
//#include "Graph.h" //from CollisionDetector.h
#include "WFObjUtils.h"
#include "Primitive.h"
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

/* Triangle/triangle intersection test routine,
* by Tomas Moller, 1997.
* See article "A Fast Triangle-Triangle Intersection Test",
* Journal of Graphics Tools, 2(2), 1997
*
* Updated June 1999: removed the divisions -- a little faster now!
* Updated October 1999: added {} to CROSS and SUB macros
*
* int NoDivTriTriIsect(float V0[3],float V1[3],float V2[3],
*                      float U0[3],float U1[3],float U2[3])
*
* parameters: vertices of triangle 1: V0,V1,V2
*             vertices of triangle 2: U0,U1,U2
* result    : returns 1 if the triangles intersect, otherwise 0
*
*/

//#include <math.h>
#define FABS(x) (fabsf(x))        /* implement as is fastest on your machine */

/* if USE_EPSILON_TEST is true then we do a check:
if |dv|<EPSILON then dv=0.0;
else no check is done (which is less robust)
*/
#define USE_EPSILON_TEST TRUE
#define EPSILON 0.000001


/* some macros */
#define CROSS(dest,v1,v2){                     \
              dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
              dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
              dest[2]=v1[0]*v2[1]-v1[1]*v2[0];}

#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])

#define SUB(dest,v1,v2){         \
            dest[0]=v1[0]-v2[0]; \
            dest[1]=v1[1]-v2[1]; \
            dest[2]=v1[2]-v2[2];}

/* sort so that a<=b */
#define SORT(a,b)       \
             if(a>b)    \
                          {          \
               float c; \
               c=a;     \
               a=b;     \
               b=c;     \
                          }


/* this edge to edge test is based on Franlin Antonio's gem:
"Faster Line Segment Intersection", in Graphics Gems III,
pp. 199-202 */
#define EDGE_EDGE_TEST(V0,U0,U1)                      \
  Bx=U0[i0]-U1[i0];                                   \
  By=U0[i1]-U1[i1];                                   \
  Cx=V0[i0]-U0[i0];                                   \
  Cy=V0[i1]-U0[i1];                                   \
  f=Ay*Bx-Ax*By;                                      \
  d=By*Cx-Bx*Cy;                                      \
  if((f>0 && d>=0 && d<=f) || (f<0 && d<=0 && d>=f))  \
    {                                                   \
    e=Ax*Cy-Ay*Cx;                                    \
    if(f>0)                                           \
        {                                                 \
      if(e>=0 && e<=f) return 1;                      \
        }                                                 \
        else                                              \
    {                                                 \
      if(e<=0 && e>=f) return 1;                      \
    }                                                 \
    }

#define EDGE_AGAINST_TRI_EDGES(V0,V1,U0,U1,U2) \
{                                              \
  float Ax,Ay,Bx,By,Cx,Cy,e,d,f;               \
  Ax=V1[i0]-V0[i0];                            \
  Ay=V1[i1]-V0[i1];                            \
  /* test edge U0,U1 against V0,V1 */          \
  EDGE_EDGE_TEST(V0,U0,U1);                    \
  /* test edge U1,U2 against V0,V1 */          \
  EDGE_EDGE_TEST(V0,U1,U2);                    \
  /* test edge U2,U1 against V0,V1 */          \
  EDGE_EDGE_TEST(V0,U2,U0);                    \
}

#define POINT_IN_TRI(V0,U0,U1,U2)           \
{                                           \
  float a,b,c,d0,d1,d2;                     \
  /* is T1 completly inside T2? */          \
  /* check if V0 is inside tri(U0,U1,U2) */ \
  a=U1[i1]-U0[i1];                          \
  b=-(U1[i0]-U0[i0]);                       \
  c=-a*U0[i0]-b*U0[i1];                     \
  d0=a*V0[i0]+b*V0[i1]+c;                   \
                                            \
  a=U2[i1]-U1[i1];                          \
  b=-(U2[i0]-U1[i0]);                       \
  c=-a*U1[i0]-b*U1[i1];                     \
  d1=a*V0[i0]+b*V0[i1]+c;                   \
                                            \
  a=U0[i1]-U2[i1];                          \
  b=-(U0[i0]-U2[i0]);                       \
  c=-a*U2[i0]-b*U2[i1];                     \
  d2=a*V0[i0]+b*V0[i1]+c;                   \
  if(d0*d1>0.0)                             \
    {                                         \
    if(d0*d2>0.0) return 1;                 \
    }                                         \
}

__host__ __device__ int coplanar_tri_tri(float N[3], float V0[3], float V1[3], float V2[3],
	float U0[3], float U1[3], float U2[3])
{
	float A[3];
	short i0, i1;
	/* first project onto an axis-aligned plane, that maximizes the area */
	/* of the triangles, compute indices: i0,i1. */
	A[0] = FABS(N[0]);
	A[1] = FABS(N[1]);
	A[2] = FABS(N[2]);
	if (A[0]>A[1])
	{
		if (A[0]>A[2])
		{
			i0 = 1;      /* A[0] is greatest */
			i1 = 2;
		}
		else
		{
			i0 = 0;      /* A[2] is greatest */
			i1 = 1;
		}
	}
	else   /* A[0]<=A[1] */
	{
		if (A[2]>A[1])
		{
			i0 = 0;      /* A[2] is greatest */
			i1 = 1;
		}
		else
		{
			i0 = 0;      /* A[1] is greatest */
			i1 = 2;
		}
	}

	/* test all edges of triangle 1 against the edges of triangle 2 */
	EDGE_AGAINST_TRI_EDGES(V0, V1, U0, U1, U2);
	EDGE_AGAINST_TRI_EDGES(V1, V2, U0, U1, U2);
	EDGE_AGAINST_TRI_EDGES(V2, V0, U0, U1, U2);

	/* finally, test if tri1 is totally contained in tri2 or vice versa */
	POINT_IN_TRI(V0, U0, U1, U2);
	POINT_IN_TRI(U0, V0, V1, V2);

	return 0;
}



#define NEWCOMPUTE_INTERVALS(VV0,VV1,VV2,D0,D1,D2,D0D1,D0D2,A,B,C,X0,X1) \
{ \
        if(D0D1>0.0f) \
                { \
                /* here we know that D0D2<=0.0 */ \
            /* that is D0, D1 are on the same side, D2 on the other or on the plane */ \
                A=VV2; B=(VV0-VV2)*D2; C=(VV1-VV2)*D2; X0=D2-D0; X1=D2-D1; \
                } \
          else if(D0D2>0.0f)\
        { \
                /* here we know that d0d1<=0.0 */ \
            A=VV1; B=(VV0-VV1)*D1; C=(VV2-VV1)*D1; X0=D1-D0; X1=D1-D2; \
        } \
          else if(D1*D2>0.0f || D0!=0.0f) \
        { \
                /* here we know that d0d1<=0.0 or that D0!=0.0 */ \
                A=VV0; B=(VV1-VV0)*D0; C=(VV2-VV0)*D0; X0=D0-D1; X1=D0-D2; \
        } \
          else if(D1!=0.0f) \
        { \
                A=VV1; B=(VV0-VV1)*D1; C=(VV2-VV1)*D1; X0=D1-D0; X1=D1-D2; \
        } \
          else if(D2!=0.0f) \
        { \
                A=VV2; B=(VV0-VV2)*D2; C=(VV1-VV2)*D2; X0=D2-D0; X1=D2-D1; \
        } \
          else \
        { \
                /* triangles are coplanar */ \
                return coplanar_tri_tri(N1,V0,V1,V2,U0,U1,U2); \
        } \
}



__host__ __device__ int NoDivTriTriIsect(float V0[3], float V1[3], float V2[3],
	float U0[3], float U1[3], float U2[3])
{
	float E1[3], E2[3];
	float N1[3], N2[3], d1, d2;
	float du0, du1, du2, dv0, dv1, dv2;
	float D[3];
	float isect1[2], isect2[2];
	float du0du1, du0du2, dv0dv1, dv0dv2;
	short index;
	float vp0, vp1, vp2;
	float up0, up1, up2;
	float bb, cc, max;

	/* compute plane equation of triangle(V0,V1,V2) */
	SUB(E1, V1, V0);
	SUB(E2, V2, V0);
	CROSS(N1, E1, E2);
	d1 = -DOT(N1, V0);
	/* plane equation 1: N1.X+d1=0 */

	/* put U0,U1,U2 into plane equation 1 to compute signed distances to the plane*/
	du0 = DOT(N1, U0) + d1;
	du1 = DOT(N1, U1) + d1;
	du2 = DOT(N1, U2) + d1;

	/* coplanarity robustness check */
#if USE_EPSILON_TEST==TRUE
	if (FABS(du0)<EPSILON) du0 = 0.0;
	if (FABS(du1)<EPSILON) du1 = 0.0;
	if (FABS(du2)<EPSILON) du2 = 0.0;
#endif
	du0du1 = du0*du1;
	du0du2 = du0*du2;

	if (du0du1>0.0f && du0du2>0.0f) /* same sign on all of them + not equal 0 ? */
		return 0;                    /* no intersection occurs */

									 /* compute plane of triangle (U0,U1,U2) */
	SUB(E1, U1, U0);
	SUB(E2, U2, U0);
	CROSS(N2, E1, E2);
	d2 = -DOT(N2, U0);
	/* plane equation 2: N2.X+d2=0 */

	/* put V0,V1,V2 into plane equation 2 */
	dv0 = DOT(N2, V0) + d2;
	dv1 = DOT(N2, V1) + d2;
	dv2 = DOT(N2, V2) + d2;

#if USE_EPSILON_TEST==TRUE
	if (FABS(dv0)<EPSILON) dv0 = 0.0;
	if (FABS(dv1)<EPSILON) dv1 = 0.0;
	if (FABS(dv2)<EPSILON) dv2 = 0.0;
#endif

	dv0dv1 = dv0*dv1;
	dv0dv2 = dv0*dv2;

	if (dv0dv1>0.0f && dv0dv2>0.0f) /* same sign on all of them + not equal 0 ? */
		return 0;                    /* no intersection occurs */

									 /* compute direction of intersection line */
	CROSS(D, N1, N2);

	/* compute and index to the largest component of D */
	max = (float)FABS(D[0]);
	index = 0;
	bb = (float)FABS(D[1]);
	cc = (float)FABS(D[2]);
	if (bb>max) max = bb, index = 1;
	if (cc>max) max = cc, index = 2;

	/* this is the simplified projection onto L*/
	vp0 = V0[index];
	vp1 = V1[index];
	vp2 = V2[index];

	up0 = U0[index];
	up1 = U1[index];
	up2 = U2[index];

	/* compute interval for triangle 1 */
	float a, b, c, x0, x1;
	NEWCOMPUTE_INTERVALS(vp0, vp1, vp2, dv0, dv1, dv2, dv0dv1, dv0dv2, a, b, c, x0, x1);

	/* compute interval for triangle 2 */
	float d, e, f, y0, y1;
	NEWCOMPUTE_INTERVALS(up0, up1, up2, du0, du1, du2, du0du1, du0du2, d, e, f, y0, y1);

	float xx, yy, xxyy, tmp;
	xx = x0*x1;
	yy = y0*y1;
	xxyy = xx*yy;

	tmp = a*xxyy;
	isect1[0] = tmp + b*x1*yy;
	isect1[1] = tmp + c*x0*yy;

	tmp = d*xxyy;
	isect2[0] = tmp + e*xx*y1;
	isect2[1] = tmp + f*xx*y0;

	SORT(isect1[0], isect1[1]);
	SORT(isect2[0], isect2[1]);

	if (isect1[1]<isect2[0] || isect2[1]<isect1[0]) return 0;
	return 1;
}


class ExactCollisionOperator
{
public:
	thrust::device_ptr<unsigned int>	objIds;
	thrust::device_ptr<unsigned int>	primIds;

	unsigned int stride;
	thrust::device_ptr<unsigned int>	adjMatrix;
	thrust::device_ptr<uint3>			triIndices;
	thrust::device_ptr<float3>			triVertices;

	ExactCollisionOperator(
		thrust::device_ptr<unsigned int>	aObjIds,
		thrust::device_ptr<unsigned int>	aPrimIds,
		unsigned int						aStride,
		thrust::device_ptr<unsigned int>	aMatrix,
		thrust::device_ptr<uint3>			aIndices,
		thrust::device_ptr<float3>			aVertices
	) :
		objIds(aObjIds),
		primIds(aPrimIds),
		stride(aStride),
		adjMatrix(aMatrix),
		triIndices(aIndices),
		triVertices(aVertices)
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
			uint3 myVtxIds = triIndices[myPrimId];
			Triangle myTri;
			myTri.vtx[0] = triVertices[myVtxIds.x];
			myTri.vtx[1] = triVertices[myVtxIds.y];
			myTri.vtx[2] = triVertices[myVtxIds.z];


			for (unsigned int otherRefId = refId + 1; otherRefId < aCellRange.y; ++otherRefId)
			{
				unsigned int otherPrimId = primIds[otherRefId];
				unsigned int otherObjId = objIds[primIds[otherRefId]];
				if (myObjId == otherObjId ||
					(myObjId == lastRecordedPair.x && otherObjId == lastRecordedPair.y ||
						myObjId == lastRecordedPair.y && otherObjId == lastRecordedPair.x))
					continue;

				uint3 otherVtxIds = triIndices[otherPrimId];
				Triangle otherTri;
				otherTri.vtx[0] = triVertices[otherVtxIds.x];
				otherTri.vtx[1] = triVertices[otherVtxIds.y];
				otherTri.vtx[2] = triVertices[otherVtxIds.z];

				float* v0 = toPtr(myTri.vtx[0]);
				float* v1 = toPtr(myTri.vtx[1]);
				float* v2 = toPtr(myTri.vtx[2]);
				
				float* u0 = toPtr(otherTri.vtx[0]);
				float* u1 = toPtr(otherTri.vtx[1]);
				float* u2 = toPtr(otherTri.vtx[2]);

				int flag = NoDivTriTriIsect(v0, v1, v2, u0, u1, u2);
				if (flag != 0)
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


Graph CollisionDetector::computeCollisionGraph(WFObject & aObj, float aRelativeThreshold)
{
	cudastd::timer timer;
	cudastd::timer intermTimer;

	//build a collision graph
	Graph result;

	if (aObj.getNumObjects() < 2u)
	{
		graphTime = intermTimer.get();
		intermTimer.cleanup();

		totalTime = timer.get();
		timer.cleanup();

		return result;
	}

	//compute scene diagonal
	float3 minBound, maxBound;
	ObjectBoundsExporter()(aObj, minBound, maxBound);
	float boundsDiagonal = len(maxBound - minBound);
	float3 res = make_float3(32.f, 32.f, 32.f);
	if (aRelativeThreshold < EPS)
	{
		const float volume = (maxBound.x - minBound.x) * (maxBound.y - minBound.y) * (maxBound.z - minBound.z);
		const float lambda = 8.f;
		const float magicConstant =
			powf(lambda * static_cast<float>(aObj.faces.size()) / volume, 0.3333333f);

		res = (maxBound - minBound) * magicConstant;
	}
	else
	{
		res = (maxBound - minBound) / (boundsDiagonal * 0.577350269f * aRelativeThreshold); //0.577350269 ~ sqrtf(3.f)
	}

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
	thrust::device_vector<uint2> trimmed_cells(grid.getNumCells());
	CellTrimmer trimmCells(objectIdPerFaceDevice.data(), grid.primitives);
	thrust::transform(grid.cells, grid.cells + grid.getNumCells(), trimmed_cells.begin(), trimmCells);

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

	if (aRelativeThreshold < EPS)
	{
		/////////////////////////////////////////////////////////////////////////////////////
		//Exact collision detection
		/////////////////////////////////////////////////////////////////////////////////////
		//compute vertex index buffer for the triangles
		thrust::host_vector<uint3> host_indices(aObj.faces.size());
		for (size_t i = 0; i < aObj.faces.size(); i++)
		{
			host_indices[i].x = (unsigned int)aObj.faces[i].vert1;
			host_indices[i].y = (unsigned int)aObj.faces[i].vert2;
			host_indices[i].z = (unsigned int)aObj.faces[i].vert3;
		}
		//copy the vertex index buffer to the device
		thrust::device_vector<uint3> device_indices(aObj.faces.size());
		thrust::copy(host_indices.begin(), host_indices.end(), device_indices.begin());

		//copy the vertex buffer to the device
		thrust::device_vector<float3> device_vertices(aObj.vertices.begin(), aObj.vertices.end());


		ExactCollisionOperator collide(
			objectIdPerFaceDevice.data(),
			grid.primitives,
			(unsigned int)aObj.objects.size(),
			adjMatrix.data(),
			device_indices.data(),
			device_vertices.data());

		//thrust::for_each(grid.cells, grid.cells + grid.getNumCells(), collide);
		thrust::for_each(trimmed_cells.begin(), trimmed_cells_end, collide);

	}
	else
	{
		/////////////////////////////////////////////////////////////////////////////////////
		//Approximate collision detection
		////////////////////////////////////////////////////////////////////////////////////

		CollisionOperator collide(
			objectIdPerFaceDevice.data(),
			grid.primitives,
			(unsigned int)aObj.objects.size(),
			adjMatrix.data());

		thrust::for_each(grid.cells, grid.cells + grid.getNumCells(), collide);

	}//end if exact/approx collision detection

	adjMatTime = intermTimer.get();
	intermTimer.start();	

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

	grid.cleanup();

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

#undef USE_EPSILON_TEST
#undef EPSILON
#undef CROSS
#undef DOT
#undef SUB
#undef SORT
#undef EDGE_EDGE_TEST
#undef EDGE_AGAINST_TRI_EDGES
#undef POINT_IN_TRI
#undef FABS
