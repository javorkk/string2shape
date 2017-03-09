#include "pch.h"
#include "CollisionGraphExporter.h"
#include "CollisionDetector.h"
#include "WFObjWriter.h"

#include "thrust/host_vector.h"

void CollisionGraphExporter::exportCollisionGraph(const char * aFileName, WFObject & aObj, Graph & aGraph) const
{
	ObjWriter output;
	output.init(aFileName);

	std::vector<float3> objCenters(aObj.objects.size(), make_float3(0.f, 0.f, 0.f));
	
	for (auto objIt = aObj.objects.begin(); objIt != aObj.objects.end(); ++objIt)
	{
		float3& midPoint = *(objCenters.begin() + (objIt - aObj.objects.begin()));
		for (int faceId = objIt->x; faceId < objIt->y; ++faceId)
		{
			midPoint += aObj.vertices[aObj.faces[faceId].vert1];
			midPoint += aObj.vertices[aObj.faces[faceId].vert2];
			midPoint += aObj.vertices[aObj.faces[faceId].vert3];
		}
		midPoint *= (0.33333f / (float)(objIt->y - objIt->x));
	}

	for (auto objIt = objCenters.begin(); objIt != objCenters.end(); ++objIt)
	{
		output.writeVertex(objIt->x, objIt->y, objIt->z);
	}

	//points (graph vertices)
	for (auto objIt = objCenters.begin(); objIt != objCenters.end(); ++objIt)
	{	
		int objId = objIt - objCenters.begin();
		output.writeObjectHeader(objId);
		
		int faceId = (aObj.objects.begin() + objId)->x;
		WFObject::Material mat = aObj.materials[aObj.faces[faceId].material];
		
		output.writeDiffuseMaterial(objId, mat.diffuseCoeff.x, mat.diffuseCoeff.y, mat.diffuseCoeff.z);
		
		output.writePointIndex(objId);
	}

	//edges
	output.writeObjectHeader(objCenters.size());
	output.writeDiffuseMaterial(objCenters.size(), 0.8f, 0.8f, 0.8f);

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
