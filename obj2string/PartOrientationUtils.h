#ifdef _MSC_VER
#pragma once
#endif

#ifndef PARTORIENTATIONUTILS_H_B86401E5_2DB4_40AA_A09C_A388E57677B8
#define PARTORIENTATIONUTILS_H_B86401E5_2DB4_40AA_A09C_A388E57677B8

#include "Algebra.h"
#include "SVD.h"
#include "WFObject.h"
#include "Graph.h"

#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>

class LocalCoordsEstimator
{
	static const bool USE_PCA = false;
public:
	uint2 * vertexRanges;
	float3*			vertexBuffer;
	double*			tmpCovMatrix;
	double*			tmpDiagonalW;
	double*			tmpMatrixV;
	double*			tmpVecRV;
	float3*			outTranslation;
	quaternion4f*	outRotation;


	LocalCoordsEstimator(
		uint2*			aRanges,
		float3*			aBuffer,
		double*			aCovMatrix,
		double*			aDiagonalW,
		double*			aMatrixV,
		double*			aVecRV,
		float3*			aOutTranslation,
		quaternion4f*	aOutRot
	) :
		vertexRanges(aRanges),
		vertexBuffer(aBuffer),
		tmpCovMatrix(aCovMatrix),
		tmpDiagonalW(aDiagonalW),
		tmpMatrixV(aMatrixV),
		tmpVecRV(aVecRV),
		outTranslation(aOutTranslation),
		outRotation(aOutRot)
	{}

	__host__ __device__	void operator()(const size_t& aId)
	{
		const unsigned int objId = (unsigned)aId;


		//Compute the mean of the vertex locations
		float3 center = make_float3(0.f, 0.f, 0.f);
		uint2 vtxRange = vertexRanges[objId];
		unsigned int vtxCount = vtxRange.y - vtxRange.x;
		float numPoints = (float)vtxCount;

		for (unsigned int vtxId = 0; vtxId < vtxCount; ++vtxId)
		{
			center += vertexBuffer[vtxRange.x + vtxId];
		}
		center /= numPoints;

		outTranslation[aId] = center;

		//Find the vertex furthest away from the center
		float3 vtx0 = center;
		float dist0 = 0.f;
		float count = 0.f;
		for (unsigned int vtxId = 0; vtxId < vtxCount; ++vtxId)
		{
			const float3 vec = vertexBuffer[vtxRange.x + vtxId];
			//const float3 delta = vec - center;
			//const float distSQR = dot(delta, delta);
			const float dist = len(vec - center);
			if (dist > dist0 && dist - dist0 > 0.001f * dist0)
			{
				vtx0 = vec;
				dist0 = dist;
				count = 1.f;
			}
			else if (fabsf(dist0 - dist) < 0.001f * dist0)
			{
				vtx0 += vec;
				count += 1.f;
			}
		}
		if (count > 1.f)
			vtx0 /= count;
		count = 0.f;
		//Find the other end of the diameter
		float3 vtx1 = vtx0;
		float diameter = 0.f;
		for (unsigned int vtxId = 0; vtxId < vtxCount; ++vtxId)
		{
			const float3 vec = vertexBuffer[vtxRange.x + vtxId];
			//const float3 delta = vec - vtx0;
			//const float distSQR = dot(delta, delta);
			const float dist = len(vec - vtx0);
			if (dist > diameter && dist - diameter > 0.001f * diameter)
			{
				vtx1 = vec;
				diameter = dist;
				count = 1.f;
			}
			else if (fabsf(diameter - dist) < 0.001f * diameter)
			{
				vtx1 += vec;
				count += 1.f;
			}
		}
		if (count > 1.f)
			vtx1 /= count;

		const float3 dir0 = ~(vtx1 - vtx0);
		//Find the vertex furthest away from the diameter
		float3 vtx2 = vtx0;
		float dist2 = 0.f;
		count = 0.f;
		for (unsigned int vtxId = 0; vtxId < vtxCount; ++vtxId)
		{
			const float3 vec = vertexBuffer[vtxRange.x + vtxId];
			const float3 delta = cross(dir0, vec - vtx0);
			const float distSQR = dot(delta, delta);
			const float distCenterSQR = dot(vec - center, vec - center);
			if (distSQR >= dist2 && distSQR - dist2 > 0.0001f * dist2)
			{
				vtx2 = vec;
				dist2 = distSQR;
				count = 1.f;
			}
		}
		if (count > 1.5f)
			vtx2 /= count;

		//vtx0 = vertexBuffer[vtxRange.x + 0];
		//vtx1 = vertexBuffer[vtxRange.x + 1];
		//vtx2 = vertexBuffer[vtxRange.x + 2];
		const float3 dir1 = ~((vtx2 - vtx0) - dir0 * dot(vtx2 - vtx0, dir0));
		const float3 dir2 = ~cross(dir0, dir1);

		float rotDet = determinant(
			dir0.x, dir1.x, dir2.x,
			dir0.y, dir1.y, dir2.y,
			dir0.z, dir1.z, dir2.z
		);

		outRotation[aId] = quaternion4f(
			dir0.x, dir1.x, dir2.x,
			dir0.y, dir1.y, dir2.y,
			dir0.z, dir1.z, dir2.z
		);

		if (USE_PCA)
		{
			//Compute covariance matrix
			double* covMat = tmpCovMatrix + aId * 3;
			for (unsigned int vtxId = 0; vtxId < vtxCount; ++vtxId)
			{
				float3 vec1 = vertexBuffer[vtxRange.x + vtxId] - center;

				covMat[0 * 3 + 0] += (double)vec1.x * vec1.x;
				covMat[1 * 3 + 0] += (double)vec1.y * vec1.x;
				covMat[2 * 3 + 0] += (double)vec1.z * vec1.x;

				covMat[0 * 3 + 1] += (double)vec1.x * vec1.y;
				covMat[1 * 3 + 1] += (double)vec1.y * vec1.y;
				covMat[2 * 3 + 1] += (double)vec1.z * vec1.y;

				covMat[0 * 3 + 2] += (double)vec1.x * vec1.z;
				covMat[1 * 3 + 2] += (double)vec1.y * vec1.z;
				covMat[2 * 3 + 2] += (double)vec1.z * vec1.z;
			}

			//Singular Value Decomposition
			double* diag = tmpDiagonalW + aId * 3;
			double* vMat = tmpMatrixV + aId * 3 * 3;
			double* tmp = tmpVecRV + aId * 3;

			svd::svdcmp(covMat, 3, 3, diag, vMat, tmp);

			const float3 col0 = make_float3((float)vMat[0], (float)vMat[1], (float)vMat[2]);
			const float3 col1 = make_float3((float)vMat[3], (float)vMat[4], (float)vMat[5]);
			const float3 col2 = make_float3((float)vMat[6], (float)vMat[7], (float)vMat[8]);

			float rotDet = determinant(
				col0.x, col1.x, col2.x,
				col0.y, col1.y, col2.y,
				col0.z, col1.z, col2.z
			);

			if (rotDet < 0.f)
			{
				vMat[0] = -vMat[0];
				vMat[1] = -vMat[1];
				vMat[2] = -vMat[2];
				rotDet = -rotDet;
			}
			if (fabsf(rotDet - 1.0f) <= 0.01f)
			{
				quaternion4f rotation(
					col0.x, col1.x, col2.x,
					col0.y, col1.y, col2.y,
					col0.z, col1.z, col2.z
				);
				outRotation[aId] = ~rotation;
			}
		}
	}

};


class TransformationExtractor
{
public:
	thrust::device_ptr<unsigned int> nodeTypes;

	thrust::device_ptr<unsigned int> outNeighborTypeKeys;
	thrust::device_ptr<unsigned int> outNeighborTypeVals;

	thrust::device_ptr<float3> translation;
	thrust::device_ptr<quaternion4f> rotation;

	thrust::device_ptr<float3> outTranslation;
	thrust::device_ptr<quaternion4f> outRotation;
	thrust::device_ptr<quaternion4f> outRotationAbs;


	TransformationExtractor(
		thrust::device_ptr<unsigned int> aNodeTypes,
		thrust::device_ptr<unsigned int> aOutNbrTypeKeys,
		thrust::device_ptr<unsigned int> aOutNbrTypeVals,
		thrust::device_ptr<float3> aTranslation,
		thrust::device_ptr<quaternion4f> aRotation,
		thrust::device_ptr<float3> aOutTranslation,
		thrust::device_ptr<quaternion4f> aOutRotation,
		thrust::device_ptr<quaternion4f> aOutRotationAbs
	) :
		nodeTypes(aNodeTypes),
		outNeighborTypeKeys(aOutNbrTypeKeys),
		outNeighborTypeVals(aOutNbrTypeVals),
		translation(aTranslation),
		rotation(aRotation),
		outTranslation(aOutTranslation),
		outRotation(aOutRotation),
		outRotationAbs(aOutRotationAbs)
	{}

	template <typename Tuple>
	__host__ __device__	void operator()(Tuple t)
	{
		const unsigned int nodeId1 = thrust::get<0>(t);
		const unsigned int nodeId2 = thrust::get<1>(t);
		const unsigned int outId = (unsigned)thrust::get<2>(t);


		outNeighborTypeKeys[outId] = nodeTypes[nodeId1];
		outNeighborTypeVals[outId] = nodeTypes[nodeId2];

		quaternion4f rot1 = rotation[nodeId1];
		outTranslation[outId] = transformVec(rot1.conjugate(), translation[nodeId2] - translation[nodeId1]);
		quaternion4f rot2 = rotation[nodeId2];
		outRotation[outId] = rot2.conjugate() * rot1;
		outRotationAbs[outId] = rot1.conjugate();
	}

};

class PartOrientationEstimator
{
	//Node a's id
	thrust::host_vector<unsigned int> mNeighborIdKeys;
	//Node b's id
	thrust::host_vector<unsigned int> mNeighborIdVals;
	//Node a's type
	thrust::host_vector<unsigned int> mNeighborTypeKeys;
	//Node b's type
	thrust::host_vector<unsigned int> mNeighborTypeVals;
	//Center of gravity of b in a's coordinate system
	thrust::host_vector<float3> mRelativeTranslation;
	//Rotates a's local coordinate frame into b's
	thrust::host_vector<quaternion4f> mRelativeRotation;
	//Rotates the canonical coordinates into a's
	thrust::host_vector<quaternion4f> mAbsoluteRotation;
	//Part sizes
	thrust::host_vector<float> mSizes;

public:

	__host__ PartOrientationEstimator()
	{}

	__host__ void init(WFObject& aObj, Graph & aGraph);

	__host__ std::vector<unsigned int> getEdges();

	__host__ std::vector<float> getOrientations();

	__host__ std::vector<float> getEdgesAndOrientations();

	__host__ std::vector<float> getEdgesTypesAndOrientations();

	__host__ quaternion4f getAbsoluteRotation(unsigned int aNodeId);

	__host__ float3 getRelativeTranslation(unsigned int aEdgeId) { return mRelativeTranslation[aEdgeId]; }

	__host__ quaternion4f getRelativeRotation(unsigned int aEdgeId) { return mRelativeRotation[aEdgeId]; }

};

#endif // PARTORIENTATIONUTILS_H_B86401E5_2DB4_40AA_A09C_A388E57677B8
