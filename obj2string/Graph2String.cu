#include "pch.h"
#include "Graph2String.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


__host__ std::string GraphToStringConverter::operator()(WFObject & aObj, Graph & aGraph)
{
	if (aObj.objects.size() != aGraph.numNodes())
	{
		std::cerr 
			<< "Number of objects " << aObj.objects.size()
			<< " and graph nodes " << aGraph.numNodes()
			<< " do not match\n";
		return std::string("");
	}

	if (aObj.materials.size() > mAlphabet.size())
	{
		std::cerr << "Too many object types " << aObj.materials.size() << "\n";
		std::cerr << "Current maximum number is " << mAlphabet.size() << "\n";
		return std::string("");
	}

	thrust::host_vector<unsigned int> nodeTypes(aGraph.numNodes(), (unsigned int)aObj.materials.size());
	for (size_t nodeId = 0; nodeId < aObj.objects.size(); ++nodeId)
	{
		size_t faceId = aObj.objects[nodeId].x;
		size_t materialId = aObj.faces[faceId].material;
		nodeTypes[nodeId] = (unsigned int)materialId;
	}

	size_t numNodes;
	thrust::device_vector<Graph::EdgeType> adjMatrixDevice;
	aGraph.toSpanningTree(adjMatrixDevice, numNodes);
	thrust::host_vector<Graph::EdgeType> adjMatrixHost(adjMatrixDevice);

	thrust::host_vector<unsigned int> cycleIds(numNodes *  numNodes, (unsigned int)-1);
	unsigned int cycleId = 0;
	for (size_t rowId = 0; rowId < numNodes; ++rowId)
	{
		for (size_t colId = rowId + 1; colId < numNodes; ++colId)
		{
			if (adjMatrixHost[colId + numNodes * rowId] == Graph::EdgeType::CYCLE)
			{
				cycleIds[colId + numNodes * rowId] = cycleId;
				cycleIds[rowId + numNodes * colId] = cycleId;
				++cycleId;
			}
		}
	}

	return std::string("");
}
