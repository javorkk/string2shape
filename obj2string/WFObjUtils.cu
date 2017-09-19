#include "pch.h"
#include "WFObjUtils.h"

#include "WFObjWriter.h"

#include <thrust/sequence.h>

__host__ void ObjectCenterExporter::operator()(
	const WFObject & aObj,
	thrust::host_vector<float3>& oObjCenters,
	thrust::host_vector<float>& oObjSizes,
	const float aSizeScaleFactor) const
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

__host__ void ObjectBoundsExporter::operator()(
	const WFObject & aObj, float3 & oMinBound, float3 & oMaxBound) const
{
	oMinBound = rep(FLT_MAX);
	oMaxBound = rep(-FLT_MAX);
	for (auto it = aObj.vertices.begin(); it != aObj.vertices.end(); ++it)
	{
		oMinBound = min(*it, oMinBound);
		oMaxBound = max(*it, oMaxBound);
	}
}

__host__ WFObject WFObjectMerger::operator()(
	const WFObject & aObj1,
	float3 aTranslation1,
	const WFObject & aObj2,
	float3 aTranslation2,
	quaternion4f aRotation2,
	thrust::host_vector<unsigned int>& aFlags1,
	thrust::host_vector<unsigned int>& aFlags2) const
{

	thrust::host_vector<float3> objCenters1;
	thrust::host_vector<float> objSizes1;

	ObjectCenterExporter()(aObj1, objCenters1, objSizes1, 0.25f);

	thrust::host_vector<float3> objCenters2;
	thrust::host_vector<float> objSizes2;

	ObjectCenterExporter()(aObj2, objCenters2, objSizes2, 0.25f);


	WFObject obj;

	//assume identical materials in aObj1 and aObj2
	obj.materials.assign(aObj1.materials.begin(), aObj1.materials.end());

	std::vector<size_t> vtxIndexMap1(aObj1.getNumVertices(), (size_t)-1);
	std::vector<size_t> vtxIndexMap2(aObj2.getNumVertices(), (size_t)-1);
	std::vector<size_t> normIndexMap1(aObj1.getNumNormals(), (size_t)-1);
	std::vector<size_t> normIndexMap2(aObj2.getNumNormals(), (size_t)-1);

	int numFaces = 0;
	for (size_t obj1Id = 0; obj1Id < aObj1.objects.size(); ++obj1Id)
	{
		if (aFlags1[obj1Id] == 0u)
			continue;
		for (int faceId = aObj1.objects[obj1Id].x; faceId < aObj1.objects[obj1Id].y; ++faceId)
		{
			WFObject::Face face(&obj);
			face.material = aObj1.faces[faceId].material;

			size_t vtxId1 = aObj1.faces[faceId].vert1;
			size_t vtxId2 = aObj1.faces[faceId].vert2;
			size_t vtxId3 = aObj1.faces[faceId].vert3;

			if (vtxIndexMap1[vtxId1] == (size_t)-1)
			{
				vtxIndexMap1[vtxId1] = obj.vertices.size();
				obj.vertices.push_back(aObj1.vertices[vtxId1] - aTranslation1);
			}
			if (vtxIndexMap1[vtxId2] == (size_t)-1)
			{
				vtxIndexMap1[vtxId2] = obj.vertices.size();
				obj.vertices.push_back(aObj1.vertices[vtxId2] - aTranslation1);
			}
			if (vtxIndexMap1[vtxId3] == (size_t)-1)
			{
				vtxIndexMap1[vtxId3] = obj.vertices.size();
				obj.vertices.push_back(aObj1.vertices[vtxId3] - aTranslation1);
			}

			face.vert1 = vtxIndexMap1[vtxId1];
			face.vert2 = vtxIndexMap1[vtxId2];
			face.vert3 = vtxIndexMap1[vtxId3];

			size_t normId1 = aObj1.faces[faceId].norm1;
			size_t normId2 = aObj1.faces[faceId].norm2;
			size_t normId3 = aObj1.faces[faceId].norm3;

			if (normIndexMap1[normId1] == (size_t)-1)
			{
				normIndexMap1[normId1] = obj.normals.size();
				obj.normals.push_back(transformVec(aRotation2.conjugate(), aObj1.normals[normId1]));
			}
			if (normIndexMap1[normId2] == (size_t)-1)
			{
				normIndexMap1[normId2] = obj.normals.size();
				obj.normals.push_back(transformVec(aRotation2.conjugate(), aObj1.normals[normId2]));
			}
			if (normIndexMap1[normId3] == (size_t)-1)
			{
				normIndexMap1[normId3] = obj.normals.size();
				obj.normals.push_back(transformVec(aRotation2.conjugate(), aObj1.normals[normId3]));
			}

			face.norm1 = normIndexMap1[normId1];
			face.norm2 = normIndexMap1[normId2];
			face.norm3 = normIndexMap1[normId3];

			obj.faces.push_back(face);
		}

		obj.objects.push_back(make_int2(numFaces, (int)obj.faces.size()));
		numFaces = (int)obj.faces.size();
	}


	for (size_t obj2Id = 0; obj2Id < aObj2.objects.size(); ++obj2Id)
	{
		if (aFlags2[obj2Id] == 0u)
			continue;
		size_t matId2 = aObj2.faces[aObj2.objects[obj2Id].x].material;
		bool overlaps = false;
		for (size_t obj1Id = 0; obj1Id < aObj1.objects.size() && !overlaps; ++obj1Id)
		{
			if (aFlags1[obj1Id] == 0u)
				continue;
			size_t matId1 = aObj1.faces[aObj1.objects[obj1Id].x].material;
			if (len((objCenters1[obj1Id] - aTranslation1) - transformVec(aRotation2, objCenters2[obj2Id] - aTranslation2)) < 0.125f * objSizes1[obj1Id] && matId1 == matId2)
				overlaps = true;
		}
		if (overlaps)
			continue;

		for (int faceId = aObj2.objects[obj2Id].x; faceId < aObj2.objects[obj2Id].y; ++faceId)
		{
			WFObject::Face face(&obj);
			face.material = aObj2.faces[faceId].material;

			size_t vtxId1 = aObj2.faces[faceId].vert1;
			size_t vtxId2 = aObj2.faces[faceId].vert2;
			size_t vtxId3 = aObj2.faces[faceId].vert3;

			if (vtxIndexMap2[vtxId1] == (size_t)-1)
			{
				vtxIndexMap2[vtxId1] = obj.vertices.size();
				obj.vertices.push_back(transformVec(aRotation2, aObj2.vertices[vtxId1] - aTranslation2));
			}
			if (vtxIndexMap2[vtxId2] == (size_t)-1)
			{
				vtxIndexMap2[vtxId2] = obj.vertices.size();
				obj.vertices.push_back(transformVec(aRotation2, aObj2.vertices[vtxId2] - aTranslation2));
			}
			if (vtxIndexMap2[vtxId3] == (size_t)-1)
			{
				vtxIndexMap2[vtxId3] = obj.vertices.size();
				obj.vertices.push_back(transformVec(aRotation2, aObj2.vertices[vtxId3] - aTranslation2));
			}

			face.vert1 = vtxIndexMap2[vtxId1];
			face.vert2 = vtxIndexMap2[vtxId2];
			face.vert3 = vtxIndexMap2[vtxId3];

			size_t normId1 = aObj2.faces[faceId].norm1;
			size_t normId2 = aObj2.faces[faceId].norm2;
			size_t normId3 = aObj2.faces[faceId].norm3;

			if (normIndexMap2[normId1] == (size_t)-1)
			{
				normIndexMap2[normId1] = obj.normals.size();
				obj.normals.push_back(aObj2.normals[normId1]);
			}
			if (normIndexMap2[normId2] == (size_t)-1)
			{
				normIndexMap2[normId2] = obj.normals.size();
				obj.normals.push_back(aObj2.normals[normId2]);
			}
			if (normIndexMap2[normId3] == (size_t)-1)
			{
				normIndexMap2[normId3] = obj.normals.size();
				obj.normals.push_back(aObj2.normals[normId3]);
			}

			face.norm1 = normIndexMap2[normId1];
			face.norm2 = normIndexMap2[normId2];
			face.norm3 = normIndexMap2[normId3];

			obj.faces.push_back(face);
		}

		obj.objects.push_back(make_int2(numFaces, (int)obj.faces.size()));
		numFaces = (int)obj.faces.size();
	}

	return obj;
}

__host__ void WFObjectFileExporter::operator()(const WFObject & aObj, const char * aFileName)
{
	ObjWriter output;
	output.init(aFileName);

	for (auto vtxIt = aObj.vertices.begin(); vtxIt != aObj.vertices.end(); ++vtxIt)
	{
		output.writeVertex(vtxIt->x, vtxIt->y, vtxIt->z);
	}

	for (auto normalIt = aObj.normals.begin(); normalIt != aObj.normals.end(); ++normalIt)
	{
		output.writeVertexNormal(normalIt->x, normalIt->y, normalIt->z);
	}

	for (auto matIt = aObj.materials.begin(); matIt != aObj.materials.end(); ++matIt)
	{
		output.writeMaterial(matIt->name.c_str(), matIt->diffuseCoeff.x, matIt->diffuseCoeff.y, matIt->diffuseCoeff.z);
	}

	for (auto objIt = aObj.objects.begin(); objIt != aObj.objects.end(); ++objIt)
	{
		int2 facesInterval = *objIt;
		if (facesInterval.x >= facesInterval.y)
			continue;

		//Assumes obj objects only consist of triangular faces
		const size_t matId = aObj.getFace(facesInterval.x).material;
		output.writeObjectHeader(objIt - aObj.objects.begin(), aObj.materials[matId].name.c_str());
		for (int face_id = facesInterval.x; face_id < facesInterval.y; face_id++)
		{
			//Assumes obj objects only consist of triangular faces
			WFObject::Face face = aObj.getFace(face_id);
			if (face.norm1 < aObj.normals.size() && face.norm2 < aObj.normals.size() && face.norm3 < aObj.normals.size())
			{
				output.writeTriangleIndices(face.vert1, face.vert2, face.vert3, face.norm1, face.norm2, face.norm3);
			}
			else
			{
				output.writeTriangleIndices((int)face.vert1, (int)face.vert2, (int)face.vert3);
			}

		}
	}

	output.cleanup();
}

__host__ void VertexBufferUnpacker::operator()(const WFObject & aObj, thrust::host_vector<uint2>& oRanges, thrust::host_vector<float3>& oVertices) const
{
	thrust::host_vector<unsigned int> nodeIds(aObj.objects.size());
	thrust::sequence(nodeIds.begin(), nodeIds.end());

	operator()(aObj, nodeIds, oRanges, oVertices);

	//oVertices.resize(aObj.faces.size() * 3u);
	//oRanges.resize(aObj.objects.size());
	//for (size_t objId = 0; objId < aObj.objects.size(); ++objId)
	//{
	//	oRanges[objId] = make_uint2(aObj.objects[objId].x * 3u, aObj.objects[objId].y * 3u);
	//	for (int faceId = aObj.objects[objId].x; faceId < aObj.objects[objId].y; ++faceId)
	//	{
	//		WFObject::Face face = aObj.faces[faceId];
	//		oVertices[faceId * 3u + 0] = aObj.vertices[aObj.faces[faceId].vert1];
	//		oVertices[faceId * 3u + 1] = aObj.vertices[aObj.faces[faceId].vert2];
	//		oVertices[faceId * 3u + 2] = aObj.vertices[aObj.faces[faceId].vert3];
	//	}
	//}
}

__host__ void VertexBufferUnpacker::operator()(const WFObject & aObj, thrust::host_vector<unsigned int>& aNodeIds, thrust::host_vector<uint2>& oRanges, thrust::host_vector<float3>& oVertices) const
{
	std::vector<float3> vertices;
	oRanges.resize(aNodeIds.size());
	for (size_t nodeId = 0; nodeId < aNodeIds.size(); ++nodeId)
	{
		unsigned int objId = aNodeIds[nodeId];
		oRanges[nodeId].x = (unsigned int)vertices.size();
		for (int faceId = aObj.objects[objId].x; faceId < aObj.objects[objId].y; ++faceId)
		{
			WFObject::Face face = aObj.faces[faceId];
			vertices.push_back(aObj.vertices[aObj.faces[faceId].vert1]);
			vertices.push_back(aObj.vertices[aObj.faces[faceId].vert2]);
			vertices.push_back(aObj.vertices[aObj.faces[faceId].vert3]);
		}
		oRanges[nodeId].y = (unsigned int)vertices.size();
	}
	
	oVertices.resize(vertices.size());
	thrust::copy(vertices.begin(), vertices.end(), oVertices.begin());
}

__host__ void ExtremeVertexUnpacker::operator()(const WFObject & aObj, thrust::host_vector<uint2>& oRanges, thrust::host_vector<float3>& oVertices) const
{
	thrust::host_vector<unsigned int> nodeIds(aObj.objects.size()); 
	thrust::sequence(nodeIds.begin(), nodeIds.end());

	operator()(aObj, nodeIds, oRanges, oVertices);
}

__host__ void ExtremeVertexUnpacker::operator()(const WFObject & aObj, thrust::host_vector<unsigned int>& aNodeIds, thrust::host_vector<uint2>& oRanges, thrust::host_vector<float3>& oVertices) const
{
	thrust::host_vector<uint2> ranges(aObj.objects.size());
	thrust::host_vector<float3> vertices(aObj.faces.size() * 3u);
	thrust::host_vector<float3> centers(aObj.faces.size());
	float3 minBound = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
	float3 maxBound = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	for (size_t objId = 0; objId < aObj.objects.size(); ++objId)
	{
		ranges[objId] = make_uint2(aObj.objects[objId].x * 3u, aObj.objects[objId].y * 3u);
		centers[objId] = make_float3(0.f, 0.f, 0.f);
		const float numVerticesRCP = 1.f / (float)(3 * (aObj.objects[objId].y - aObj.objects[objId].x));
		for (int faceId = aObj.objects[objId].x; faceId < aObj.objects[objId].y; ++faceId)
		{
			WFObject::Face face = aObj.faces[faceId];
			vertices[faceId * 3u + 0] = aObj.vertices[aObj.faces[faceId].vert1];
			vertices[faceId * 3u + 1] = aObj.vertices[aObj.faces[faceId].vert2];
			vertices[faceId * 3u + 2] = aObj.vertices[aObj.faces[faceId].vert3];
			centers[objId] += numVerticesRCP *  (vertices[faceId * 3u + 0] + vertices[faceId * 3u + 1] + vertices[faceId * 3u + 2]);
			
			minBound = min(minBound, vertices[faceId * 3u + 0]);
			minBound = min(minBound, vertices[faceId * 3u + 1]);
			minBound = min(minBound, vertices[faceId * 3u + 2]);

			maxBound = max(maxBound, vertices[faceId * 3u + 0]);
			maxBound = max(maxBound, vertices[faceId * 3u + 1]);
			maxBound = max(maxBound, vertices[faceId * 3u + 2]);

		}
	}
	const float diagonalSQR = dot(maxBound - minBound, maxBound - minBound);

	std::vector<float3> extremeVertices;
	oRanges.resize(aNodeIds.size());

	for (size_t nodeId = 0; nodeId < aNodeIds.size(); ++nodeId)
	{
		unsigned int objId = aNodeIds[nodeId];
		oRanges[nodeId].x = (unsigned int)extremeVertices.size();
		for (unsigned int vtxId1 = ranges[objId].x; vtxId1 < ranges[objId].y; ++vtxId1)
		{
			//unsigned int vtxId0 = vtxId1;
			//float maxDistSQR = 0.f;
			//for (unsigned int vtxId2 = ranges[objId].x; vtxId2 < ranges[objId].y; ++vtxId2)
			//{
			//	if (vtxId2 == vtxId1)
			//		continue;
			//	const float distSQR = dot(vertices[vtxId1] - vertices[vtxId2], vertices[vtxId1] - vertices[vtxId2]);
			//	if (distSQR > maxDistSQR)
			//	{
			//		maxDistSQR = distSQR;
			//		vtxId0 = vtxId2;
			//	}
			//}
			float3 pseudoRadius = ~(centers[objId] - vertices[vtxId1]);
			bool valid = true;
			for (unsigned int vtxId2 = ranges[objId].x; vtxId2 < ranges[objId].y && valid; ++vtxId2)
			{
				if (vtxId2 == vtxId1)
					continue;
				if (dot(pseudoRadius, ~(vertices[vtxId2] - vertices[vtxId1])) < 0.001f)
					valid = false;
			}

			for (unsigned int vtxId2 = oRanges[nodeId].x; vtxId2 < extremeVertices.size() && valid; ++vtxId2)
			{
				if (dot(extremeVertices[vtxId2] - vertices[vtxId1], extremeVertices[vtxId2] - vertices[vtxId1]) < 0.001f * diagonalSQR)
					valid = false;
			}

			if (valid)
			{
				float3 vtx = vertices[vtxId1];
				extremeVertices.push_back(vtx);
			}
		}
		oRanges[nodeId].y = (unsigned int)extremeVertices.size();

	}

	oVertices.resize(extremeVertices.size());
	thrust::copy(extremeVertices.begin(), extremeVertices.end(), oVertices.begin());
}

__host__ bool WFObjectSanityTest::operator()(const WFObject & aObj) const
{
	return aObj.getNumVertices() > 0u && aObj.getNumFaces() > 0u && aObj.getNumObjects() > 0u && aObj.getNumMaterials() > 1u;
}
