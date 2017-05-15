#include "pch.h"
#include "WFObjUtils.h"

__host__ void ObjectCenterExporter::operator()(WFObject & aObj, thrust::host_vector<float3>& oObjCenters, thrust::host_vector<float>& oObjSizes, const float aSizeScaleFactor)
{
	oObjCenters = thrust::host_vector<float3>(aObj.objects.size(), make_float3(0.f, 0.f, 0.f));
	oObjSizes = thrust::host_vector<float>(aObj.objects.size(), 1.f);

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

__host__ WFObject WFObjectMerger::operator()(const WFObject & aObj1, float3 aTranslation1, const WFObject & aObj2, float3 aTranslation2, quaternion4f aRotation2, thrust::host_vector<unsigned int>& aFlags1, thrust::host_vector<unsigned int>& aFlags2)
{
	WFObject obj;

	obj.vertices.resize(aObj1.getNumVertices() + aObj2.getNumVertices());
	for (size_t vtxId = 0u; vtxId < aObj1.getNumVertices(); ++vtxId)
	{
		obj.vertices[vtxId] = aObj1.vertices[vtxId] - aTranslation1;
	}

	for (size_t vtxId = 0u; vtxId < aObj2.getNumVertices(); ++vtxId)
	{
		obj.vertices[vtxId + aObj1.getNumVertices()] = transformVec(aRotation2,aObj2.vertices[vtxId] - aTranslation2);
	}


	obj.normals.resize(aObj1.getNumNormals() + aObj2.getNumNormals());
	for (size_t nId = 0u; nId < aObj1.getNumNormals(); ++nId)
	{
		obj.normals[nId] = aObj1.normals[nId];
	}

	for (size_t nId = 0u; nId < aObj2.getNumNormals(); ++nId)
	{
		obj.normals[nId + aObj1.getNumNormals()] = transformVec(aRotation2, aObj2.normals[nId]);
	}

	//assume identical materials
	obj.materials.assign(aObj1.materials.begin(), aObj1.materials.end());
	
	int numFaces = 0;	
	for (size_t obj1Id = 0; obj1Id < aObj1.objects.size(); ++obj1Id)
	{
		if (aFlags1[obj1Id] == 0u)
			continue;
		for (int faceId = aObj1.objects[obj1Id].x; faceId < aObj1.objects[obj1Id].y; ++faceId)
		{
			WFObject::Face face(&obj);
			face.material = aObj1.faces[faceId].material;

			face.vert1 = aObj1.faces[faceId].vert1;
			face.vert2 = aObj1.faces[faceId].vert2;
			face.vert3 = aObj1.faces[faceId].vert3;

			face.norm1 = aObj1.faces[faceId].norm1;
			face.norm2 = aObj1.faces[faceId].norm2;
			face.norm3 = aObj1.faces[faceId].norm3;

			face.tex1 = aObj1.faces[faceId].tex1;
			face.tex2 = aObj1.faces[faceId].tex2;
			face.tex3 = aObj1.faces[faceId].tex3;

			obj.faces.push_back(face);
		}

		obj.objects.push_back(make_int2(numFaces, (int)obj.faces.size()));
		numFaces = (int)obj.faces.size();
	}


	for (size_t obj2Id = 0; obj2Id < aObj2.objects.size(); ++obj2Id)
	{
		if (aFlags2[obj2Id] == 0u)
			continue;
		for (int faceId = aObj2.objects[obj2Id].x; faceId < aObj2.objects[obj2Id].y; ++faceId)
		{
			WFObject::Face face(&obj);
			face.material = aObj2.faces[faceId].material;

			face.vert1 = aObj2.faces[faceId].vert1 + aObj1.getNumVertices();
			face.vert2 = aObj2.faces[faceId].vert2 + aObj1.getNumVertices();
			face.vert3 = aObj2.faces[faceId].vert3 + aObj1.getNumVertices();

			face.norm1 = aObj2.faces[faceId].norm1 + aObj1.getNumNormals();
			face.norm2 = aObj2.faces[faceId].norm2 + aObj1.getNumNormals();
			face.norm3 = aObj2.faces[faceId].norm3 + aObj1.getNumNormals();

			face.tex1 = aObj2.faces[faceId].tex1 + aObj1.getNumTexCoords();
			face.tex2 = aObj2.faces[faceId].tex2 + aObj1.getNumTexCoords();
			face.tex3 = aObj2.faces[faceId].tex3 + aObj1.getNumTexCoords();

			obj.faces.push_back(face);
		}

		obj.objects.push_back(make_int2(numFaces, (int)obj.faces.size()));
		numFaces = (int)obj.faces.size();
	}

	return obj;
}
