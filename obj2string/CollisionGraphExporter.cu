#include "pch.h"
#include "CollisionGraphExporter.h"
#include "CollisionDetector.h"
#include "WFObjWriter.h"
#include "WFObjUtils.h"

#include "DebugUtils.h"
#include "Timer.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

void CollisionGraphExporter::exportCollisionGraph(const char * aFilePath, WFObject & aObj, Graph & aGraph)
{
	cudastd::timer timer;

	size_t numNodes;
	thrust::device_vector<Graph::EdgeType> adjMatrixDevice;
	aGraph.toSpanningTree(adjMatrixDevice, numNodes);

	if (numNodes != aObj.objects.size())
		std::cerr << "Collision graph error! Expected " << aObj.objects.size() << " nodes, got " << numNodes << ".\n";

#ifdef _DEBUG
	outputDeviceVector("adjacency matrix: ", adjMatrixDevice);
#endif

	thrust::host_vector<Graph::EdgeType> adjMatrixHost(adjMatrixDevice);

	std::string fileName(aFilePath);
	if (fileName.find_last_of("/\\") == std::string::npos)
		fileName = fileName.substr(0, fileName.size() - 5);
	else
		fileName = fileName.substr(fileName.find_last_of("/\\") + 1, fileName.size() - fileName.find_last_of("/\\") - 5);

	std::string objDir = getDirName(aFilePath);
	std::string graphFilePath = objDir + fileName + "_coll_graph";
	ObjWriter output;
	output.init(graphFilePath.c_str());

	//std::cerr << "Exporting collision graph to " << graphFilePath << ".obj ...\n";

	thrust::host_vector<float3> objCenters;// (aObj.objects.size(), make_float3(0.f, 0.f, 0.f));
	thrust::host_vector<float> objSizes;// (aObj.objects.size(), 1.f);

	ObjectCenterExporter()(aObj, objCenters, objSizes, 0.3333f);

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

	for (size_t objId = 0u; objId < objCenters.size(); ++objId)
	{
		output.writeVertex(objCenters[objId].x, objCenters[objId].y, objCenters[objId].z);
	}

	//cubes (graph vertices)
	auto sizeIt = objSizes.begin();
	for (int objId = 0; objId < objCenters.size(); ++objId, ++sizeIt)
	{
		output.writeVertex(objCenters[objId].x - *sizeIt, objCenters[objId].y - *sizeIt, objCenters[objId].z - *sizeIt); //000
		output.writeVertex(objCenters[objId].x + *sizeIt, objCenters[objId].y - *sizeIt, objCenters[objId].z - *sizeIt); //100
		output.writeVertex(objCenters[objId].x - *sizeIt, objCenters[objId].y + *sizeIt, objCenters[objId].z - *sizeIt); //010
		output.writeVertex(objCenters[objId].x + *sizeIt, objCenters[objId].y + *sizeIt, objCenters[objId].z - *sizeIt); //110
		output.writeVertex(objCenters[objId].x - *sizeIt, objCenters[objId].y - *sizeIt, objCenters[objId].z + *sizeIt); //001
		output.writeVertex(objCenters[objId].x + *sizeIt, objCenters[objId].y - *sizeIt, objCenters[objId].z + *sizeIt); //101
		output.writeVertex(objCenters[objId].x - *sizeIt, objCenters[objId].y + *sizeIt, objCenters[objId].z + *sizeIt); //011
		output.writeVertex(objCenters[objId].x + *sizeIt, objCenters[objId].y + *sizeIt, objCenters[objId].z + *sizeIt); //111
		
		output.writeObjectHeader(objId);

		int faceId = (aObj.objects.begin() + objId)->x;
		WFObject::Material mat = aObj.materials[aObj.faces[faceId].material];

		output.writeDiffuseMaterial(objId, mat.diffuseCoeff.x * (float)M_PI, mat.diffuseCoeff.y * (float)M_PI, mat.diffuseCoeff.z * (float)M_PI);
		int offset = (int)objCenters.size() + objId * 8;

		//xy quads
		output.writeTriangleIndices(offset + 0, offset + 3, offset + 1);
		output.writeTriangleIndices(offset + 0, offset + 2, offset + 3);
		output.writeTriangleIndices(offset + 4, offset + 5, offset + 7);
		output.writeTriangleIndices(offset + 4, offset + 7, offset + 6);

		//yz quads
		output.writeTriangleIndices(offset + 0, offset + 6, offset + 2);
		output.writeTriangleIndices(offset + 0, offset + 4, offset + 6);
		output.writeTriangleIndices(offset + 1, offset + 3, offset + 7);
		output.writeTriangleIndices(offset + 1, offset + 7, offset + 5);

		//xz quads
		output.writeTriangleIndices(offset + 0, offset + 1, offset + 5);
		output.writeTriangleIndices(offset + 0, offset + 5, offset + 4);
		output.writeTriangleIndices(offset + 2, offset + 7, offset + 3);
		output.writeTriangleIndices(offset + 2, offset + 6, offset + 7);

		//output.writePointIndex(objId);
	}


	//spanning tree edges
	output.writeObjectHeader((int)objCenters.size());
	output.writeDiffuseMaterial((int)objCenters.size(), 0.6f, 0.6f, 0.6f);

	thrust::host_vector<int> edgesA(aGraph.adjacencyKeys);
	thrust::host_vector<int> edgesB(aGraph.adjacencyVals);

	for (int edgeId = 0; edgeId < edgesA.size(); ++edgeId)
	{
		if (edgesA[edgeId] > edgesB[edgeId])
			continue; // do not output duplicate edges
		int edgeLinearId = edgesA[edgeId] + (int)numNodes * edgesB[edgeId];
		if (adjMatrixHost[edgeLinearId] != Graph::EdgeType::SPANNING_TREE)
			continue;
		output.writeLineIndices(edgesA[edgeId], edgesB[edgeId]);
	}

	//removed cycle edges
	output.writeObjectHeader((int)objCenters.size() + 1);
	output.writeDiffuseMaterial((int)objCenters.size() + 1, 0.6f, 0.7f, 0.0f);
	for (int edgeId = 0; edgeId < edgesA.size(); ++edgeId)
	{
		if (edgesA[edgeId] > edgesB[edgeId])
			continue; // do not output duplicate edges
		int edgeLinearId = edgesA[edgeId] + (int)numNodes * edgesB[edgeId];
		if (adjMatrixHost[edgeLinearId] != Graph::EdgeType::CYCLE)
			continue;
		output.writeLineIndices(edgesA[edgeId], edgesB[edgeId]);
	}

	output.cleanup();

	totalTime = timer.get();
	timer.cleanup();
}

__host__ void CollisionGraphExporter::exportSubGraph(const char * aFilePath, WFObject & aObj, Graph & aGraph, size_t aId, const thrust::host_vector<unsigned int>& aNodeFlags)
{
	cudastd::timer timer;

	size_t numNodes;
	thrust::device_vector<Graph::EdgeType> adjMatrixDevice;
	aGraph.toSpanningTree(adjMatrixDevice, numNodes);

	if (numNodes != aObj.objects.size())
		std::cerr << "Collision sub-graph error! Expected " << aObj.objects.size() << " nodes, got " << numNodes << ".\n";

	if (aNodeFlags.size() != aObj.objects.size())
		std::cerr << "Collision sub-graph error! Expected " << aNodeFlags.size() << " node flags, got " << numNodes << ".\n";

#ifdef _DEBUG
	outputDeviceVector("adjacency matrix: ", adjMatrixDevice);
#endif

	thrust::host_vector<Graph::EdgeType> adjMatrixHost(adjMatrixDevice);

	std::string fileName(aFilePath);
	if (fileName.find_last_of("/\\") == std::string::npos)
		fileName = fileName.substr(0, fileName.size() - 5);
	else
		fileName = fileName.substr(fileName.find_last_of("/\\") + 1, fileName.size() - fileName.find_last_of("/\\") - 5);

	std::string objDir = getDirName(aFilePath);
	std::string graphFilePath = objDir + fileName + "_sub_graph_" + itoa((int)aId);
	ObjWriter output;
	output.init(graphFilePath.c_str());

	//std::cerr << "Exporting collision graph to " << graphFilePath << ".obj ...\n";

	thrust::host_vector<float3> objCenters;// (aObj.objects.size(), make_float3(0.f, 0.f, 0.f));
	thrust::host_vector<float> objSizes;// (aObj.objects.size(), 1.f);

	ObjectCenterExporter()(aObj, objCenters, objSizes, 0.3333f);

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

	for (size_t objId = 0u; objId < objCenters.size(); ++objId)
	{
		output.writeVertex(objCenters[objId].x, objCenters[objId].y, objCenters[objId].z);
	}

	//cubes (graph vertices)
	auto sizeIt = objSizes.begin();
	for (int objId = 0; objId < objCenters.size(); ++objId, ++sizeIt)
	{
		output.writeVertex(objCenters[objId].x - *sizeIt, objCenters[objId].y - *sizeIt, objCenters[objId].z - *sizeIt); //000
		output.writeVertex(objCenters[objId].x + *sizeIt, objCenters[objId].y - *sizeIt, objCenters[objId].z - *sizeIt); //100
		output.writeVertex(objCenters[objId].x - *sizeIt, objCenters[objId].y + *sizeIt, objCenters[objId].z - *sizeIt); //010
		output.writeVertex(objCenters[objId].x + *sizeIt, objCenters[objId].y + *sizeIt, objCenters[objId].z - *sizeIt); //110
		output.writeVertex(objCenters[objId].x - *sizeIt, objCenters[objId].y - *sizeIt, objCenters[objId].z + *sizeIt); //001
		output.writeVertex(objCenters[objId].x + *sizeIt, objCenters[objId].y - *sizeIt, objCenters[objId].z + *sizeIt); //101
		output.writeVertex(objCenters[objId].x - *sizeIt, objCenters[objId].y + *sizeIt, objCenters[objId].z + *sizeIt); //011
		output.writeVertex(objCenters[objId].x + *sizeIt, objCenters[objId].y + *sizeIt, objCenters[objId].z + *sizeIt); //111
		
		if (aNodeFlags[objId] == 0u)
			continue;

		output.writeObjectHeader(objId);

		int faceId = (aObj.objects.begin() + objId)->x;
		WFObject::Material mat = aObj.materials[aObj.faces[faceId].material];

		output.writeDiffuseMaterial(objId, mat.diffuseCoeff.x * (float)M_PI, mat.diffuseCoeff.y * (float)M_PI, mat.diffuseCoeff.z * (float)M_PI);
		int offset = (int)objCenters.size() + objId * 8;

		//xy quads
		output.writeTriangleIndices(offset + 0, offset + 3, offset + 1);
		output.writeTriangleIndices(offset + 0, offset + 2, offset + 3);
		output.writeTriangleIndices(offset + 4, offset + 5, offset + 7);
		output.writeTriangleIndices(offset + 4, offset + 7, offset + 6);

		//yz quads
		output.writeTriangleIndices(offset + 0, offset + 6, offset + 2);
		output.writeTriangleIndices(offset + 0, offset + 4, offset + 6);
		output.writeTriangleIndices(offset + 1, offset + 3, offset + 7);
		output.writeTriangleIndices(offset + 1, offset + 7, offset + 5);

		//xz quads
		output.writeTriangleIndices(offset + 0, offset + 1, offset + 5);
		output.writeTriangleIndices(offset + 0, offset + 5, offset + 4);
		output.writeTriangleIndices(offset + 2, offset + 7, offset + 3);
		output.writeTriangleIndices(offset + 2, offset + 6, offset + 7);

		//output.writePointIndex(objId);
	}


	//spanning tree edges
	output.writeObjectHeader((int)objCenters.size());
	output.writeDiffuseMaterial((int)objCenters.size(), 0.6f, 0.6f, 0.6f);

	thrust::host_vector<int> edgesA(aGraph.adjacencyKeys);
	thrust::host_vector<int> edgesB(aGraph.adjacencyVals);

	for (int edgeId = 0; edgeId < edgesA.size(); ++edgeId)
	{
		if (edgesA[edgeId] > edgesB[edgeId])
			continue; // do not output duplicate edges
		int edgeLinearId = edgesA[edgeId] + (int)numNodes * edgesB[edgeId];
		if (adjMatrixHost[edgeLinearId] != Graph::EdgeType::SPANNING_TREE)
			continue;
		if (aNodeFlags[edgesA[edgeId]] == 0u && aNodeFlags[edgesB[edgeId]] == 0u)
			continue;
		output.writeLineIndices(edgesA[edgeId], edgesB[edgeId]);
	}

	//removed cycle edges
	output.writeObjectHeader((int)objCenters.size() + 1);
	output.writeDiffuseMaterial((int)objCenters.size() + 1, 0.6f, 0.7f, 0.0f);
	for (int edgeId = 0; edgeId < edgesA.size(); ++edgeId)
	{
		if (edgesA[edgeId] > edgesB[edgeId])
			continue; // do not output duplicate edges
		int edgeLinearId = edgesA[edgeId] + (int)numNodes * edgesB[edgeId];
		if (adjMatrixHost[edgeLinearId] != Graph::EdgeType::CYCLE)
			continue;
		if (aNodeFlags[edgesA[edgeId]] == 0u && aNodeFlags[edgesB[edgeId]] == 0u)
			continue;
		output.writeLineIndices(edgesA[edgeId], edgesB[edgeId]);
	}

	output.cleanup();

	totalTime = timer.get();
	timer.cleanup();
}

__host__ void CollisionGraphExporter::stats()
{
	std::cerr << "Collision graph exported in " << totalTime << "ms\n";
}
