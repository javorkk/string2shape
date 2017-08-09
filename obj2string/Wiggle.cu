#include "pch.h"
#include "Wiggle.h"

#include <deque>

#include "WFObjUtils.h"
#include "SVD.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

class LocalCoordsEstimator
{
public:
	uint2*			vertexRanges;
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
		thrust::raw_pointer_cast(vertexRangesDevice.data()),
		thrust::raw_pointer_cast(vertexBufferDevice.data()),
		thrust::raw_pointer_cast(tmpCovMatrix.data()),
		thrust::raw_pointer_cast(tmpDiagonalW.data()),
		thrust::raw_pointer_cast(tmpMatrixV.data()),
		thrust::raw_pointer_cast(tmpVecRV.data()),
		thrust::raw_pointer_cast(outTranslation.data()),
		thrust::raw_pointer_cast(outRotation.data())
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
		mNeighborTypeVals = thrust::host_vector<unsigned int>(neighborTypeVals);
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

	//sort by node type
	thrust::sort_by_key(
		mNeighborTypeKeys.begin(),
		mNeighborTypeKeys.end(),
		thrust::make_zip_iterator(thrust::make_tuple(mNeighborTypeVals.begin(), mRelativeTranslation.begin(), mRelativeRotation.begin()))
		);

	//setup search intervals for each node type
	mIntervals.resize(aObj.materials.size() + 1u, 0u);
	for (size_t i = 0u; i < mNeighborTypeKeys.size() - 1u; ++i)
	{
		if (mNeighborTypeKeys[i] < mNeighborTypeKeys[i + 1u])
		{
			mIntervals[mNeighborTypeKeys[i + 1]] = (unsigned)i + 1u;
		}
	}
	//last element
	if (mNeighborTypeKeys.size() > 0u)
		mIntervals[mNeighborTypeKeys[mNeighborTypeKeys.size() - 1u]] = (unsigned)mNeighborTypeKeys.size();

	//fill gaps due to missing node types
	for (size_t i = 1u; i < mIntervals.size(); ++i)
	{
		mIntervals[i] = std::max(mIntervals[i - 1u], mIntervals[i]);
	}
}

__host__ void Wiggle::fixRelativeTransformations(WFObject & aObj, Graph & aGraph)
{
	//Unpack and upload the vertex buffer
	thrust::host_vector<uint2> vertexRangesHost;
	thrust::host_vector<float3> vertexBufferHost;

	VertexBufferUnpacker unpackVertices;
	unpackVertices(aObj, vertexRangesHost, vertexBufferHost);

	thrust::device_vector<uint2> vertexRangesDevice(vertexRangesHost);
	thrust::device_vector<float3> vertexBufferDevice(vertexBufferHost);


	size_t numNodes = aObj.objects.size();
	thrust::host_vector<unsigned int> visited(numNodes, 0u);
	thrust::host_vector<unsigned int> intervalsHost(aGraph.intervals);
	thrust::host_vector<unsigned int> adjacencyValsHost(aGraph.adjacencyVals);

	//Extract and upload node type information
	thrust::host_vector<unsigned int> nodeTypesHost(aGraph.numNodes(), (unsigned int)aObj.materials.size());
	for (size_t nodeId = 0; nodeId < aObj.objects.size(); ++nodeId)
	{
		size_t faceId = aObj.objects[nodeId].x;
		size_t materialId = aObj.faces[faceId].material;
		nodeTypesHost[nodeId] = (unsigned int)materialId;
	}

	std::deque<unsigned int> frontier;
	frontier.push_back(0u);
	visited[0u] = 1u;
	while (!frontier.empty())
	{
		const unsigned int nodeId = frontier.front();
		frontier.pop_front();
		
		processNeighbors(
			aObj,
			nodeId,
			visited,
			intervalsHost,
			adjacencyValsHost,
			nodeTypesHost);

		for (unsigned int nbrId = intervalsHost[nodeId]; nbrId < intervalsHost[nodeId + 1]; ++nbrId)
		{
			const unsigned int nodeId = adjacencyValsHost[nbrId];
			if (visited[nodeId] == 0u)
			{
				frontier.push_back(nodeId);
				visited[nodeId] = 1u;
			}
		}

	}

}

__host__ void Wiggle::processNeighbors(
	WFObject&							aObj,
	unsigned int						aObjId,
	thrust::host_vector<unsigned int>&	visited,
	thrust::host_vector<unsigned int>&	intervalsHost,
	thrust::host_vector<unsigned int>&	adjacencyValsHost,
	thrust::host_vector<unsigned int>&	nodeTypeIds)
{
	const unsigned int nbrCount = intervalsHost[aObjId + 1u] - intervalsHost[aObjId];

	if (nbrCount == 0)
		return;

	const unsigned int nodeCount = nbrCount + 1u;
	thrust::host_vector<unsigned int> nodeIds(nodeCount, aObjId);
	thrust::copy(nodeIds.begin() + 1u, nodeIds.end(), adjacencyValsHost.begin() + intervalsHost[aObjId]);
	
	unsigned int vtxCount = 0u;
	for (unsigned int i = 0; i < nodeIds.size(); i++)
	{
		vtxCount += 3 * (aObj.objects[nodeIds[i]].y - aObj.objects[nodeIds[i]].x);
	}

	//Unpack the vertex buffer
	thrust::host_vector<float3> vertexBufferHost(vtxCount);
	thrust::host_vector<uint2> vtxRanges(nodeCount);
	unsigned int currentVtxId = 0u;
	for (unsigned int i = 0; i < nodeIds.size(); i++)
	{
		vtxRanges[i].x = currentVtxId;
		for (int faceId = aObj.objects[nodeIds[i]].x; faceId < aObj.objects[nodeIds[i]].y; ++faceId)
		{
			WFObject::Face face = aObj.faces[faceId];
			vertexBufferHost[currentVtxId++] = aObj.vertices[aObj.faces[faceId].vert1];
			vertexBufferHost[currentVtxId++] = aObj.vertices[aObj.faces[faceId].vert2];
			vertexBufferHost[currentVtxId++] = aObj.vertices[aObj.faces[faceId].vert3];
		}
		vtxRanges[i].y = currentVtxId;
	}


	//Use PCA to compute local coordiante system for each object
	thrust::host_vector<float3> translations(nodeCount);
	thrust::host_vector<quaternion4f> rotations(nodeCount);
	thrust::host_vector<double> tmpCovMatrix(nodeCount * 3 * 3, 0.f);
	thrust::host_vector<double> tmpDiagonalW(nodeCount * 3);
	thrust::host_vector<double> tmpMatrixV(nodeCount * 3 * 3);
	thrust::host_vector<double> tmpVecRV(nodeCount * 3);

	LocalCoordsEstimator estimateT(
		thrust::raw_pointer_cast(vtxRanges.data()),
		thrust::raw_pointer_cast(vertexBufferHost.data()),
		thrust::raw_pointer_cast(tmpCovMatrix.data()),
		thrust::raw_pointer_cast(tmpDiagonalW.data()),
		thrust::raw_pointer_cast(tmpMatrixV.data()),
		thrust::raw_pointer_cast(tmpVecRV.data()),
		thrust::raw_pointer_cast(translations.data()),
		thrust::raw_pointer_cast(rotations.data())
	);

	thrust::counting_iterator<size_t> first(0u);
	thrust::counting_iterator<size_t> last(nodeCount);

	thrust::for_each(first, last, estimateT);

	for (unsigned int i = 1; i < nodeIds.size(); i++)
	{
		const unsigned int nbrNodeId = nodeIds[i];
		if (visited[nbrNodeId])
			continue;

		const unsigned int nodeId1 = nodeIds[0];
		const unsigned int nodeId2 = nbrNodeId;

		const unsigned int typeId1 = nodeTypeIds[nodeId1];
		const unsigned int typeId2 = nodeTypeIds[nodeId2];

		quaternion4f rot = rotations[nodeId1];
		quaternion4f irot = rot.conjugate();
		float3 relativeT = transformVec(irot, translations[nodeId2] - translations[nodeId1]);
		quaternion4f relativeR = rotations[nodeId2] * irot;

		float3 bestT;
		quaternion4f bestR;

		findBestMatch(typeId1, typeId2, relativeT, relativeR, bestT, bestR);

		if (magnitudeSQR(relativeR * bestR.conjugate()) < angleTolerance)
			continue;

		transformObj(aObj, nodeId2, -translations[i], relativeR.conjugate(), bestR, bestT, irot, translations[0]);
	}

}

__host__ void Wiggle::findBestMatch(
	unsigned int		aTypeId1,
	unsigned int		aTypeId2,
	const float3&		aTranslation,
	const quaternion4f&	aRotation,
	float3&				oTranslation,
	quaternion4f&		oRotation)
{
	float bestSpatialDist = FLT_MAX;
	for (unsigned int id = mIntervals[aTypeId1]; id < mIntervals[aTypeId1 + 1]; id++)
	{
		if (mNeighborTypeVals[id] != aTypeId2)
			continue;
		const float3 delta = mRelativeTranslation[id] - aTranslation;
		const float currentSpatialDist = dot(delta, delta);
		if (currentSpatialDist < bestSpatialDist)
		{
			bestSpatialDist = currentSpatialDist;
			oTranslation = mRelativeTranslation[id];
			oRotation = mRelativeRotation[id];
		}
	}
}

__host__ void Wiggle::transformObj(
	WFObject & aObj,
	unsigned int aObjId,
	const float3 & aTranslation0toB,
	const quaternion4f & aRotationBtoA,
	const quaternion4f & aRotationAtoC,
	const float3 & aTranslationAtoC,
	const quaternion4f & aRotationAto0,
	const float3 & aTranslationAto0)
{
	thrust::host_vector<unsigned int> processed(aObj.getNumVertices(), 0u);
	for (int faceId = aObj.objects[aObjId].x; faceId < aObj.objects[aObjId].y; ++faceId)
	{
		WFObject::Face face = aObj.faces[faceId];
		size_t vtxId1 = aObj.faces[faceId].vert1;
		size_t vtxId2 = aObj.faces[faceId].vert2;
		size_t vtxId3 = aObj.faces[faceId].vert3;
		if (processed[vtxId1] == 0u)
		{
			processed[vtxId1] = 1u;
			float3 vtx = aObj.vertices[vtxId1];
			aObj.vertices[vtxId1] = transformVtx(vtx, aTranslation0toB, aRotationBtoA, aRotationAtoC, aTranslationAtoC, aRotationAto0, aTranslationAto0);
		}
		if (processed[vtxId2] == 0u)
		{
			processed[vtxId2] = 1u;
			float3 vtx = aObj.vertices[vtxId2];
			aObj.vertices[vtxId2] = transformVtx(vtx, aTranslation0toB, aRotationBtoA, aRotationAtoC, aTranslationAtoC, aRotationAto0, aTranslationAto0);

		}
		if (processed[vtxId3] == 0u)
		{
			processed[vtxId3] = 1u;
			float3 vtx = aObj.vertices[vtxId2];
			aObj.vertices[vtxId2] = transformVtx(vtx, aTranslation0toB, aRotationBtoA, aRotationAtoC, aTranslationAtoC, aRotationAto0, aTranslationAto0);
		}
	}

}

__host__ __device__ float3 Wiggle::transformVtx(
	const float3 & aVtx,
	const float3 & aTranslation0toB,
	const quaternion4f & aRotationBtoA,
	const quaternion4f & aRotationAtoC,
	const float3 & aTranslationAtoC,
	const quaternion4f & aRotationAto0,
	const float3 & aTranslationAto0)
{
	const float3 t_b_to_0 = aVtx + aTranslation0toB;
	const float3 r_b_to_a = transformVec(aRotationBtoA, t_b_to_0);
	const float3 r_a_to_c = transformVec(aRotationAtoC, r_b_to_a);
	const float3 t_a_to_c = r_a_to_c + aTranslationAtoC;
	const float3 r_a_to_0 = transformVec(aRotationAto0, t_a_to_c);
	const float3 t_a_to_0 = r_a_to_0 + aTranslationAto0;
	return t_a_to_0;
}
