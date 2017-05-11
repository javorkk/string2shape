#include "pch.h"
#include "WFObjUtils.h"

__host__ void ObjectCenterExporter::operator()(WFObject & aObj, std::vector<float3>& oObjCenters, std::vector<float>& oObjSizes, const float aSizeScaleFactor)
{
	oObjCenters = std::vector<float3>(aObj.objects.size(), make_float3(0.f, 0.f, 0.f));
	oObjSizes = std::vector<float>(aObj.objects.size(), 1.f);

	for (auto objIt = aObj.objects.begin(); objIt != aObj.objects.end(); ++objIt)
	{
		float3 minBound = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
		float3 maxBound = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

		float3& midPoint = *(oObjCenters.begin() + (objIt - aObj.objects.begin()));
		for (int faceId = objIt->x; faceId < objIt->y; ++faceId)
		{
			float3 vtx1 = aObj.vertices[aObj.faces[faceId].vert1];
			float3 vtx2 = aObj.vertices[aObj.faces[faceId].vert2];
			float3 vtx3 = aObj.vertices[aObj.faces[faceId].vert3];
			midPoint += vtx1;
			midPoint += vtx2;
			midPoint += vtx3;
			minBound = min(minBound, vtx1);
			minBound = min(minBound, vtx2);
			minBound = min(minBound, vtx3);
			maxBound = max(maxBound, vtx1);
			maxBound = max(maxBound, vtx2);
			maxBound = max(maxBound, vtx3);

		}
		midPoint *= (0.33333f / (float)(objIt->y - objIt->x));
		float3 objSize = maxBound - minBound;

		auto sizeIt = (oObjSizes.begin() + (objIt - aObj.objects.begin()));
		*sizeIt = len(objSize) * aSizeScaleFactor;
	}
}

__host__ void ObjectBoundsExporter::operator()(WFObject & aObj, float3 & oMinBound, float3 & oMaxBound)
{
	oMinBound = rep(FLT_MAX);
	oMaxBound = rep(-FLT_MAX);
	for (auto it = aObj.vertices.begin(); it != aObj.vertices.end(); ++it)
	{
		oMinBound = min(*it, oMinBound);
		oMaxBound = max(*it, oMaxBound);
	}
}
