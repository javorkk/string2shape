#include "pch.h"
#include "CollisionGraphExporter.h"
#include "CollisionDetector.h"
#include "WFObjWriter.h"

#include <thrust/host_vector.h>

void CollisionGraphExporter::exportCollisionGraph(const char * aFileName, WFObject & aObj, Graph & aGraph) const
{
	ObjWriter output;
	output.init(aFileName);

	std::vector<float3> objCenters(aObj.objects.size(), make_float3(0.f, 0.f, 0.f));
	std::vector<float> objSizes(aObj.objects.size(), 1.f);

	for (auto objIt = aObj.objects.begin(); objIt != aObj.objects.end(); ++objIt)
	{
		float3 minBound = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
		float3 maxBound = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

		float3& midPoint = *(objCenters.begin() + (objIt - aObj.objects.begin()));
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

		auto sizeIt = (objSizes.begin() + (objIt - aObj.objects.begin()));
		*sizeIt = len(objSize) * 0.3333f * 0.1f; 
	}

	for (auto objIt = objCenters.begin(); objIt != objCenters.end(); ++objIt)
	{
		output.writeVertex(objIt->x, objIt->y, objIt->z);
	}

	//cubes (graph vertices)
	auto sizeIt = objSizes.begin();
	for (auto objIt = objCenters.begin(); objIt != objCenters.end(); ++objIt, ++sizeIt)
	{
		output.writeVertex(objIt->x - *sizeIt, objIt->y - *sizeIt, objIt->z - *sizeIt); //000
		output.writeVertex(objIt->x + *sizeIt, objIt->y - *sizeIt, objIt->z - *sizeIt); //100
		output.writeVertex(objIt->x - *sizeIt, objIt->y + *sizeIt, objIt->z - *sizeIt); //010
		output.writeVertex(objIt->x + *sizeIt, objIt->y + *sizeIt, objIt->z - *sizeIt); //110
		output.writeVertex(objIt->x - *sizeIt, objIt->y - *sizeIt, objIt->z + *sizeIt); //001
		output.writeVertex(objIt->x + *sizeIt, objIt->y - *sizeIt, objIt->z + *sizeIt); //101
		output.writeVertex(objIt->x - *sizeIt, objIt->y + *sizeIt, objIt->z + *sizeIt); //011
		output.writeVertex(objIt->x + *sizeIt, objIt->y + *sizeIt, objIt->z + *sizeIt); //111

		int objId = int(objIt - objCenters.begin());
		output.writeObjectHeader(objId);

		int faceId = (aObj.objects.begin() + objId)->x;
		WFObject::Material mat = aObj.materials[aObj.faces[faceId].material];

		output.writeDiffuseMaterial(objId, mat.diffuseCoeff.x, mat.diffuseCoeff.y, mat.diffuseCoeff.z);
		int offset = (int)objCenters.size() + objId * 8;

		//output.writeTriangleIndices(offset + 0, offset + 1, offset + 3);
		//output.writeTriangleIndices(offset + 0, offset + 3, offset + 2);
		//output.writeTriangleIndices(offset + 1, offset + 6, offset + 2);
		//output.writeTriangleIndices(offset + 6, offset + 1, offset + 5);
		//output.writeTriangleIndices(offset + 4, offset + 6, offset + 7);
		//output.writeTriangleIndices(offset + 4, offset + 7, offset + 5);
		//output.writeTriangleIndices(offset + 2, offset + 6, offset + 3);
		//output.writeTriangleIndices(offset + 6, offset + 7, offset + 3);
		//output.writeTriangleIndices(offset + 0, offset + 3, offset + 7);
		//output.writeTriangleIndices(offset + 0, offset + 7, offset + 4);
		//output.writeTriangleIndices(offset + 0, offset + 5, offset + 1);
		//output.writeTriangleIndices(offset + 0, offset + 4, offset + 5);



		//xy quads
		output.writeTriangleIndices(offset + 0, offset + 1, offset + 3);
		output.writeTriangleIndices(offset + 0, offset + 3, offset + 2);
		output.writeTriangleIndices(offset + 4, offset + 7, offset + 5);
		output.writeTriangleIndices(offset + 4, offset + 6, offset + 7);

		//yz quads
		output.writeTriangleIndices(offset + 0, offset + 2, offset + 6);
		output.writeTriangleIndices(offset + 0, offset + 6, offset + 4);
		output.writeTriangleIndices(offset + 1, offset + 7, offset + 3);
		output.writeTriangleIndices(offset + 1, offset + 5, offset + 7);

		//xz quads
		output.writeTriangleIndices(offset + 0, offset + 1, offset + 5);
		output.writeTriangleIndices(offset + 0, offset + 5, offset + 4);
		output.writeTriangleIndices(offset + 2, offset + 7, offset + 3);
		output.writeTriangleIndices(offset + 2, offset + 6, offset + 7);

		//output.writePointIndex(objId);
	}


	//edges
	output.writeObjectHeader((int)objCenters.size());
	output.writeDiffuseMaterial((int)objCenters.size(), 0.8f, 0.8f, 0.8f);

	thrust::host_vector<int> edgesA(aGraph.adjacencyKeys);
	thrust::host_vector<int> edgesB(aGraph.adjacencyVals);

	for (int edgeId = 0; edgeId < edgesA.size(); ++edgeId)
	{
		if (edgesA[edgeId] > edgesB[edgeId])
			continue; // do not output duplicate edges
		output.writeLineIndices(edgesA[edgeId], edgesB[edgeId]);
	}


	output.cleanup();
}
