#ifdef _MSC_VER
#pragma once
#endif

#ifndef WFOBJUTILS_H_2CF02B80_6A49_4293_B9A9_077F0C467114
#define WFOBJUTILS_H_2CF02B80_6A49_4293_B9A9_077F0C467114

#include "WFObject.h"
#include <thrust/host_vector.h>

class ObjectCenterExporter
{
public:
	__host__ void operator()(
		const WFObject& aObj,
		thrust::host_vector<float3>& oObjCenters,
		thrust::host_vector<float>& oObjSizes,
		const float aSizeScaleFactor = 1.f) const;
};

class ObjectBoundsExporter
{
public:
	__host__ void operator()(
		const WFObject& aObj, float3& oMinBound, float3& oMaxBound) const;
};

class WFObjectMerger
{
public:
	//Union of aObj1 - aTranslation1 and aRotation * (aObj2 - aTranslation2)
	//Only objects with non-zero flags participate
	__host__ WFObject operator()(
		const WFObject& aObj1,
		float3 aTranslation1,
		const WFObject& aObj2,
		float3 aTranslation2,
		quaternion4f aRotation2,
		thrust::host_vector<unsigned int>& aFlags1,
		thrust::host_vector<unsigned int>& aFlags2) const;
};

class WFObjectFileExporter
{
public:
	__host__ void operator()(const WFObject& aObj, const char* aFileName);
};

class VertexBufferUnpacker
{
public:
	__host__ void operator()(
		const WFObject& aObj,
		thrust::host_vector<uint2>& oRanges,
		thrust::host_vector<float3>& oVertices) const;

	__host__ void operator()(
		const WFObject& aObj,
		thrust::host_vector<unsigned int>& aNodeIds,
		thrust::host_vector<uint2>& oRanges,
		thrust::host_vector<float3>& oVertices) const;
};

class ExtremeVertexUnpacker
{
public:
	__host__ void operator()(
		const WFObject& aObj,
		thrust::host_vector<uint2>& oRanges,
		thrust::host_vector<float3>& oVertices) const;

	__host__ void operator()(
		const WFObject& aObj,
		thrust::host_vector<unsigned int>& aNodeIds,
		thrust::host_vector<uint2>& oRanges,
		thrust::host_vector<float3>& oVertices) const;

};

class WFObjectSanityTest
{
public:
	__host__ bool operator()(const WFObject& aObj) const;
};

#endif //WFOBJUTILS_H_2CF02B80_6A49_4293_B9A9_077F0C467114