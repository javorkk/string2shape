#include "pch.h"
#include "Wiggle.h"


#include "WFObjUtils.h"
#include "SVD.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

class LocalCoordsEstimator
{
public:
	thrust::device_ptr<uint2> vertexRanges;
	thrust::device_ptr<float3> vertexBuffer;


	thrust::device_ptr<double> tmpCovMatrix;
	thrust::device_ptr<double> tmpDiagonalW;
	thrust::device_ptr<double> tmpMatrixV;
	thrust::device_ptr<double> tmpVecRV;

	thrust::device_ptr<float3> outTranslation;
	thrust::device_ptr<quaternion4f> outRotation;


	LocalCoordsEstimator(
		thrust::device_ptr<uint2> aRanges,
		thrust::device_ptr<float3> aBuffer,
		thrust::device_ptr<double> aCovMatrix,
		thrust::device_ptr<double> aDiagonalW,
		thrust::device_ptr<double> aMatrixV,
		thrust::device_ptr<double> aVecRV,
		thrust::device_ptr<float3> aOutTranslation,
		thrust::device_ptr<quaternion4f> aOutRot
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


		//Compute the means of the border node locations
		float3 center = make_float3(0.f, 0.f, 0.f);
		uint2 vtxRange = vertexRanges[objId];
		unsigned int vtxCount = vtxRange.y - vtxRange.x;
		float numPoints = (float)vtxCount;

		for (unsigned int vtxId = 0; vtxId < vtxCount; ++vtxId)
		{
			center += vertexBuffer[vtxRange.x + vtxId];
		}
		center /= numPoints;
		
		//Compute covariance matrix
		double* covMat = thrust::raw_pointer_cast(tmpCovMatrix + aId * 3 * 3);
		for (unsigned int vtxId = 0; vtxId < vtxCount; ++vtxId)
		{
			float3 vec1 = vertexBuffer[vtxRange.x + vtxId];

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
		double* diag = thrust::raw_pointer_cast(tmpDiagonalW + aId * 3);
		double* vMat = thrust::raw_pointer_cast(tmpMatrixV + aId * 3 * 3);
		double* tmp = thrust::raw_pointer_cast(tmpVecRV + aId * 3);

		svd::svdcmp(covMat, 3, 3, diag, vMat, tmp);

		//Rotation is V * transpose(U)		
		for (unsigned int row = 0; row < 3; ++row)
		{
			for (unsigned int col = 0; col < 3; ++col)
			{
				tmp[col] =
					vMat[row * 3 + 0] * covMat[col * 3 + 0] +
					vMat[row * 3 + 1] * covMat[col * 3 + 1] +
					vMat[row * 3 + 2] * covMat[col * 3 + 2];
			}
			vMat[row * 3 + 0] = tmp[0];
			vMat[row * 3 + 1] = tmp[1];
			vMat[row * 3 + 2] = tmp[2];
		}


		double rotDet = determinantd(
			vMat[0], vMat[3], vMat[6],
			vMat[1], vMat[4], vMat[7],
			vMat[2], vMat[5], vMat[8]
		);

		//if (rotDet < 0.f)
		//{
		//	vMat[0] = -vMat[0];
		//	vMat[1] = -vMat[1];
		//	vMat[2] = -vMat[2];
		//	rotDet = -rotDet;
		//}


		quaternion4f rotation(
			(float)vMat[0], (float)vMat[3], (float)vMat[6],
			(float)vMat[1], (float)vMat[4], (float)vMat[7],
			(float)vMat[2], (float)vMat[5], (float)vMat[8]
		);
		outTranslation[aId] = center;
		outRotation[aId] = rotation;
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


	TransformationExtractor(
		thrust::device_ptr<unsigned int> aNodeTypes,
		thrust::device_ptr<unsigned int> aOutNbrTypeKeys,
		thrust::device_ptr<unsigned int> aOutNbrTypeVals,
		thrust::device_ptr<float3> aTranslation,
		thrust::device_ptr<quaternion4f> aRotation,
		thrust::device_ptr<float3> aOutTranslation,
		thrust::device_ptr<quaternion4f> aOutRotation
	) :
		nodeTypes(aNodeTypes),
		outNeighborTypeKeys(aOutNbrTypeKeys),
		outNeighborTypeVals(aOutNbrTypeVals),
		translation(aTranslation),
		rotation(aRotation),
		outTranslation(aOutTranslation),
		outRotation(aOutRotation)
	{}

	template <typename Tuple>
	__host__ __device__	void operator()(Tuple t)
	{
		const unsigned int nodeId1 = thrust::get<0>(t);
		const unsigned int nodeId2 = thrust::get<1>(t);
		const unsigned int outId = (unsigned)thrust::get<2>(t);


		outNeighborTypeKeys[outId] = nodeTypes[nodeId1];
		outNeighborTypeVals[outId] = nodeTypes[nodeId2];

		quaternion4f rot = rotation[nodeId1];
		quaternion4f irot = rot.conjugate();
		outTranslation[outId] = transformVec(irot, translation[nodeId2] - translation[nodeId1]);
		outRotation[outId] = rotation[nodeId2] * irot;
	}

};


__host__ void Wiggle::init(WFObject & aObj, Graph & aGraph)
{
	//Unpack and upload the vertex buffer
	thrust::host_vector<uint2> vertexRangesHost;
	thrust::host_vector<float3> vertexBufferHost;

	VertexBufferUnpacker unpackVertices;
	unpackVertices(aObj, vertexRangesHost, vertexBufferHost);

	thrust::device_vector<uint2> vertexRangesDevice(vertexRangesHost);
	thrust::device_vector<float3> vertexBufferDevice(vertexBufferHost);


	//Use PCA to compute local coordiante system for each object
	thrust::device_vector<float3> outTranslation(aObj.getNumObjects());
	thrust::device_vector<quaternion4f> outRotation(aObj.getNumObjects());

	thrust::device_vector<double> tmpCovMatrix(aObj.getNumObjects() * 3 * 3, 0.f);
	thrust::device_vector<double> tmpDiagonalW(aObj.getNumObjects() * 3);
	thrust::device_vector<double> tmpMatrixV(aObj.getNumObjects() * 3 * 3);
	thrust::device_vector<double> tmpVecRV(aObj.getNumObjects() * 3);

	LocalCoordsEstimator estimateT(
		vertexRangesDevice.data(),
		vertexBufferDevice.data(),
		tmpCovMatrix.data(),
		tmpDiagonalW.data(),
		tmpMatrixV.data(),
		tmpVecRV.data(),
		outTranslation.data(),
		outRotation.data()
	);

	thrust::counting_iterator<size_t> first(0u);
	thrust::counting_iterator<size_t> last(aObj.getNumObjects());

	thrust::for_each(first, last, estimateT);

	//Extract and upload node type information
	thrust::host_vector<unsigned int> nodeTypesHost(aGraph.numNodes(), (unsigned int)aObj.materials.size());
	for (size_t nodeId = 0; nodeId < aObj.objects.size(); ++nodeId)
	{
		size_t faceId = aObj.objects[nodeId].x;
		size_t materialId = aObj.faces[faceId].material;
		nodeTypesHost[nodeId] = (unsigned int)materialId;
	}
	thrust::device_vector<unsigned int> nodeTypes(nodeTypesHost);

	thrust::device_vector<unsigned int> neighborTypeKeys(aGraph.numEdges() * 2u);
	thrust::device_vector<unsigned int> neighborTypeVals(aGraph.numEdges() * 2u);
	thrust::device_vector<float3> relativeTranslation(aGraph.numEdges() * 2u);
	thrust::device_vector<quaternion4f> relativeRotation(aGraph.numEdges() * 2u);

	TransformationExtractor extractRelativeT(
		nodeTypes.data(),
		neighborTypeKeys.data(),
		neighborTypeVals.data(),
		outTranslation.data(),
		outRotation.data(),
		relativeTranslation.data(),
		relativeRotation.data()
	);

	thrust::counting_iterator<size_t> lastEdge(aGraph.numEdges() * 2u);

	thrust::for_each(
		thrust::make_zip_iterator(thrust::make_tuple(aGraph.adjacencyKeys.begin(), aGraph.adjacencyVals.begin(), first)),
		thrust::make_zip_iterator(thrust::make_tuple(aGraph.adjacencyKeys.end(), aGraph.adjacencyVals.end(), lastEdge)),
		extractRelativeT);

	if(mNeighborTypeKeys.size() == 0u)
	{ 
		//first call of init
		mNeighborTypeKeys = thrust::host_vector<unsigned int>(neighborTypeKeys);
		mNeighborTypeVals = thrust::host_vector<unsigned int>(mNeighborTypeVals);
		mRelativeTranslation = thrust::host_vector<float3>(relativeTranslation);
		mRelativeRotation = thrust::host_vector<quaternion4f>(relativeRotation);
	}
	else
	{
		//init already called, append new data
		size_t oldCount = mNeighborTypeKeys.size();
		mNeighborTypeKeys.resize(oldCount + neighborTypeKeys.size());
		mNeighborTypeVals.resize(oldCount + neighborTypeVals.size());
		mRelativeTranslation.resize(oldCount + relativeTranslation.size());
		mRelativeRotation.resize(oldCount + relativeRotation.size());

		thrust::copy(neighborTypeKeys.begin(), neighborTypeKeys.end(), mNeighborTypeKeys.begin() + oldCount);
		thrust::copy(neighborTypeVals.begin(), neighborTypeVals.end(), mNeighborTypeVals.begin() + oldCount);
		thrust::copy(relativeTranslation.begin(), relativeTranslation.end(), mRelativeTranslation.begin() + oldCount);
		thrust::copy(relativeRotation.begin(), relativeRotation.end(), mRelativeRotation.begin() + oldCount);
	}
}

__host__ void Wiggle::refine(WFObject & aObj, Graph & aGraph)
{
}
