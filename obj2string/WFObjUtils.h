#ifdef _MSC_VER
#pragma once
#endif

#ifndef WFOBJUTILS_H_2CF02B80_6A49_4293_B9A9_077F0C467114
#define WFOBJUTILS_H_2CF02B80_6A49_4293_B9A9_077F0C467114

#include "WFObject.h"

class ObjectCenterExporter
{
public:
	__host__ void operator()(WFObject& aObj, std::vector<float3>& oObjCenters, std::vector<float>& oObjSizes, const float aSizeScaleFactor = 1.f);
};

class ObjectBoundsExporter
{
public:
	__host__ void operator()(WFObject& aObj, float3& oMinBound, float3& oMaxBound);
};

#endif //WFOBJUTILS_H_2CF02B80_6A49_4293_B9A9_077F0C467114