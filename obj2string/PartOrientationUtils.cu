#include "pch.h"
#include "PartOrientationUtils.h"

#include "WFObjUtils.h"
#include "DebugUtils.h"

#include <thrust/device_vector.h>
#include <thrust/copy.h>

__host__ void PartOrientationEstimator::init(WFObject & aObj, Graph & aGraph)
{
	//Unpack and upload the vertex buffer
	thrust::host_vector<uint2> vertexRangesHost;
	thrust::host_vector<float3> vertexBufferHost;

	VertexBufferUnpacker unpackVertices;
	unpackVertices(aObj, vertexRangesHost, vertexBufferHost);

	thrust::device_vector<uint2> vertexRangesDevice(vertexRangesHost);
	thrust::device_vector<float3> vertexBufferDevice(vertexBufferHost);


	//#ifdef _DEBUG
	//	outputDeviceVector("vertex ranges: ", vertexRangesDevice);
	//	outputDeviceVector("vertex buffer: ", vertexBufferDevice);
	//#endif

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

	//#ifdef _DEBUG
	//	outputDeviceVector("translations: ", outTranslation);
	//	outputDeviceVector("rotations: ", outRotation);
	//#endif

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
	thrust::device_vector<quaternion4f> absoluteRotation(aGraph.numEdges() * 2u);

	TransformationExtractor extractRelativeT(
		nodeTypes.data(),
		neighborTypeKeys.data(),
		neighborTypeVals.data(),
		outTranslation.data(),
		outRotation.data(),
		relativeTranslation.data(),
		relativeRotation.data(),
		absoluteRotation.data()
	);

	thrust::counting_iterator<size_t> lastEdge(aGraph.numEdges() * 2u);

	thrust::for_each(
		thrust::make_zip_iterator(thrust::make_tuple(aGraph.adjacencyKeys.begin(), aGraph.adjacencyVals.begin(), first)),
		thrust::make_zip_iterator(thrust::make_tuple(aGraph.adjacencyKeys.end(), aGraph.adjacencyVals.end(), lastEdge)),
		extractRelativeT);

	mNeighborIdKeys = thrust::host_vector<unsigned int>(aGraph.adjacencyKeys);
	mNeighborIdVals = thrust::host_vector<unsigned int>(aGraph.adjacencyVals);
	mNeighborTypeKeys = thrust::host_vector<unsigned int>(neighborTypeKeys);
	mNeighborTypeVals = thrust::host_vector<unsigned int>(neighborTypeVals);
	mRelativeTranslation = thrust::host_vector<float3>(relativeTranslation);
	mRelativeRotation = thrust::host_vector<quaternion4f>(relativeRotation);
	mAbsoluteRotation = thrust::host_vector<quaternion4f>(absoluteRotation);

	thrust::host_vector<float3> objCenters;
	thrust::host_vector<float> objSizes;
	ObjectCenterExporter()(aObj, objCenters, objSizes);

	mSizes = thrust::host_vector<float>(mNeighborIdKeys.size(), 0.f);
	for (size_t i = 0; i < mSizes.size(); ++i)
		mSizes[i] = objSizes[mNeighborIdVals[i]];

	mNeighborConfigurations.resize(aObj.getNumObjects());
	thrust::host_vector<unsigned int> intervalsHost(aGraph.intervals);
	for (size_t nodeId = 0; nodeId < aObj.getNumObjects(); ++nodeId)
	{
		for (size_t edgeId1 = intervalsHost[nodeId]; edgeId1 < intervalsHost[nodeId + 1]; ++edgeId1)
		{
			for (size_t edgeId0 = intervalsHost[nodeId]; edgeId0 < edgeId1; ++edgeId0 )
			{
				PariwiseNeighborConfiguration current;
				current.typeA = nodeTypes[nodeId];
				current.typeNbrB0 = mNeighborTypeVals[edgeId0];
				current.typeNbrB1 = mNeighborTypeVals[edgeId1];
				current.dist = len(objCenters[mNeighborIdVals[edgeId0]] - objCenters[mNeighborIdVals[edgeId1]]);
				current.size = objSizes[nodeId];
				mNeighborConfigurations[nodeId].push_back(current);
			}
		}
	}

#ifdef _DEBUG
	outputHostVector("translations: ", mRelativeTranslation);
	outputHostVector("rotations: ", mRelativeRotation);
#endif

}

__host__ std::vector<unsigned int> PartOrientationEstimator::getEdges()
{
	std::vector<unsigned int> result(mNeighborIdKeys.size() * 2);
	for (size_t i = 0u; i < mNeighborIdKeys.size(); ++i)
	{
		result[2u * i + 0u] = mNeighborIdKeys[i];
		result[2u * i + 1u] = mNeighborIdVals[i];
	}
	return result;
}

__host__ std::vector<float> PartOrientationEstimator::getOrientations()
{
	std::vector<float> result(mNeighborIdKeys.size() * 7);
	for (size_t i = 0u; i < mNeighborIdKeys.size(); ++i)
	{
		result[7u * i + 0u] = mRelativeTranslation[i].x;
		result[7u * i + 1u] = mRelativeTranslation[i].y;
		result[7u * i + 2u] = mRelativeTranslation[i].z;
			   
		result[7u * i + 3u] = mRelativeRotation[i].x;
		result[7u * i + 4u] = mRelativeRotation[i].y;
		result[7u * i + 5u] = mRelativeRotation[i].z;
		result[7u * i + 6u] = mRelativeRotation[i].w;
	}
	return result;
}

__host__ std::vector<float> PartOrientationEstimator::getEdgesAndOrientations()
{
	std::vector<float> result(mNeighborIdKeys.size() * 9);
	for (size_t i = 0u; i < mNeighborIdKeys.size(); ++i)
	{
		result[9u * i + 0u] = (float)mNeighborIdKeys[i] + 0.1f;
		result[9u * i + 1u] = (float)mNeighborIdVals[i] + 0.1f;

		result[9u * i + 2u] = mRelativeTranslation[i].x;
		result[9u * i + 3u] = mRelativeTranslation[i].y;
		result[9u * i + 4u] = mRelativeTranslation[i].z;

		result[9u * i + 5u] = mRelativeRotation[i].x;
		result[9u * i + 6u] = mRelativeRotation[i].y;
		result[9u * i + 7u] = mRelativeRotation[i].z;
		result[9u * i + 8u] = mRelativeRotation[i].w;
	}
	return result;
}

__host__ std::vector<float> PartOrientationEstimator::getEdgesTypesAndOrientations()
{
	std::vector<float> result(mNeighborIdKeys.size() * 16u);
	for (size_t i = 0u; i < mNeighborIdKeys.size(); ++i)
	{
		result[16u * i +  0u] = (float)mNeighborIdKeys[i] + 0.1f;
		result[16u * i +  1u] = (float)mNeighborIdVals[i] + 0.1f;

		result[16u * i +  2u] = (float)mNeighborTypeKeys[i] + 0.1f;
		result[16u * i +  3u] = (float)mNeighborTypeVals[i] + 0.1f;


		result[16u * i +  4u] = mRelativeTranslation[i].x;
		result[16u * i +  5u] = mRelativeTranslation[i].y;
		result[16u * i +  6u] = mRelativeTranslation[i].z;

		result[16u * i +  7u] = mRelativeRotation[i].x;
		result[16u * i +  8u] = mRelativeRotation[i].y;
		result[16u * i +  9u] = mRelativeRotation[i].z;
		result[16u * i + 10u] = mRelativeRotation[i].w;

		result[16u * i + 11u] = mSizes[i];

		result[16u * i + 12u] = mAbsoluteRotation[i].x;
		result[16u * i + 13u] = mAbsoluteRotation[i].y;
		result[16u * i + 14u] = mAbsoluteRotation[i].z;
		result[16u * i + 15u] = mAbsoluteRotation[i].w;

	}
	return result;
}

__host__ quaternion4f PartOrientationEstimator::getAbsoluteRotation(unsigned int aNodeId)
{
	for (size_t i = 0; i < mNeighborIdKeys.size(); ++i)
		if (mNeighborIdKeys[i] == aNodeId)
			return mAbsoluteRotation[i];

	return make_quaternion4f(0.f, 0.f, 0.f, 1.f);
}

__host__ bool PartOrientationEstimator::checkNeighborConfiguration(const std::vector<PariwiseNeighborConfiguration> aConfiguration) const
{
	if (aConfiguration.size() == 0u)
		return true;

	for (size_t bucketId = 0u; bucketId < mNeighborConfigurations.size(); ++bucketId)
	{
		const std::vector<PariwiseNeighborConfiguration> currentConfiguration = mNeighborConfigurations[bucketId];
		if (currentConfiguration.size() < aConfiguration.size())
			continue;
		if (currentConfiguration[0].typeA != aConfiguration[0].typeA)
			continue;

		bool foundMismatch = false;
		for (size_t itemId = 0u; itemId < aConfiguration.size() && !foundMismatch; ++itemId)
		{
			bool itemFound = false;
			for (size_t itemId1 = 0u; itemId1 < currentConfiguration.size() && !itemFound; ++itemId1)
				if (currentConfiguration[itemId1] == aConfiguration[itemId])
					itemFound = true;

			if (!itemFound)
				foundMismatch = true;
		}

		if (!foundMismatch)
			return true;
	}

	return false;
}
