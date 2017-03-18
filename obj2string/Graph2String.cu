#include "pch.h"
#include "Graph2String.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__host__ std::string GraphToStringConverter::depthFirstTraverse(
	unsigned int nodeId,
	thrust::host_vector<unsigned int>& visited,
	unsigned int parentId,
	thrust::host_vector<unsigned int>& intervalsHost,
	thrust::host_vector<unsigned int>& adjacencyValsHost,
	thrust::host_vector<Graph::EdgeType>& adjacencyMatrixType,
	thrust::host_vector<unsigned int>& cycleIds,
	thrust::host_vector<unsigned int>& nodeTypeIds)
{
	const unsigned int numNodes = (unsigned int)intervalsHost.size() - 1;
	visited[nodeId] = 1u;
	std::string result = mAlphabet[nodeTypeIds[nodeId]];
	std::string cycleLables;
	std::string lastSubtree;
	std::string subtreeStrings;

	// Recur for all the vertices adjacent to this vertex
	for (unsigned int nbrId = intervalsHost[nodeId]; nbrId < intervalsHost[nodeId + 1]; ++nbrId)
	{
		unsigned int nbrNodeId = adjacencyValsHost[nbrId];
		if (nbrNodeId == parentId)
			continue;
		if (adjacencyMatrixType[nbrNodeId + numNodes * nodeId] == Graph::EdgeType::CYCLE)
		{
			if (cycleLables.empty())
			{
				cycleLables.append(itoa((int)cycleIds[nbrNodeId + numNodes * nodeId]));
			}
			else
			{
				cycleLables.append(mNumberSeparator);
				cycleLables.append(itoa((int)cycleIds[nbrNodeId + numNodes * nodeId]));
			}
		}
		if (adjacencyMatrixType[nbrNodeId + numNodes * nodeId] != Graph::EdgeType::SPANNING_TREE)
			continue;
		// If an adjacent is not visited, then recur for that adjacent
		if (visited[nbrNodeId] == 0)
		{
			std::string subtreeStr = depthFirstTraverse(
				nbrNodeId,
				visited,
				nodeId,
				intervalsHost,
				adjacencyValsHost,
				adjacencyMatrixType,
				cycleIds,
				nodeTypeIds
			);

			if (!lastSubtree.empty())
			{
				subtreeStrings.append(mBranchStart);
				subtreeStrings.append(lastSubtree);
				subtreeStrings.append(mBranchEnd);
			}
			lastSubtree = subtreeStr;
		}
	}

	result.append(cycleLables);
	result.append(subtreeStrings);
	result.append(lastSubtree);

	return result;
}

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

	thrust::host_vector<unsigned int> visited(numNodes, 0);
	thrust::host_vector<unsigned int> intervalsHost(aGraph.intervals);
	thrust::host_vector<unsigned int> adjacencyValsHost(aGraph.adjacencyVals);

	std::string result = depthFirstTraverse(
		0u,
		visited,
		(unsigned int)-1,
		intervalsHost,
		adjacencyValsHost,
		adjMatrixHost,
		cycleIds,
		nodeTypes);

	return result;
}
