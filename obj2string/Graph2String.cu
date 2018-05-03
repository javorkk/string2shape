#include "pch.h"
#include "Graph2String.h"
#include "WFObjUtils.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include <deque>

__host__ std::pair< std::string, std::string > GraphToStringConverter::depthFirstTraverse(
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
	
	std::pair< std::string, std::string > result = std::make_pair(mAlphabet[nodeTypeIds[nodeId]], itoa(nodeId) + " ");
	std::string cycleLables;
	std::pair< std::string, std::string > lastSubtree;
	std::pair< std::string, std::string > subtreeStrings;

	std::vector<unsigned int> permutedIds(adjacencyValsHost.begin() + intervalsHost[nodeId], adjacencyValsHost.begin() + intervalsHost[nodeId + 1]);

	std::shuffle(permutedIds.begin(), permutedIds.end(), mRNG);

	// Recur for all the vertices adjacent to this vertex
	for (unsigned int nbrId = intervalsHost[nodeId]; nbrId < intervalsHost[nodeId + 1]; ++nbrId)
	{
		//unsigned int nbrNodeId = adjacencyValsHost[nbrId];
		unsigned int nbrNodeId = permutedIds[nbrId - intervalsHost[nodeId]];
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
			std::pair< std::string, std::string > subtreeStrPair = depthFirstTraverse(
				nbrNodeId,
				visited,
				nodeId,
				intervalsHost,
				adjacencyValsHost,
				adjacencyMatrixType,
				cycleIds,
				nodeTypeIds
			);

			if (!lastSubtree.first.empty())
			{
				subtreeStrings.first.append(mBranchStart);
				subtreeStrings.first.append(lastSubtree.first);
				subtreeStrings.first.append(mBranchEnd);

				subtreeStrings.second.append(lastSubtree.second);
			}
			lastSubtree = subtreeStrPair;
		}
	}

	result.first.append(cycleLables);
	result.first.append(subtreeStrings.first);
	result.first.append(lastSubtree.first);

	result.second.append(subtreeStrings.second);
	result.second.append(lastSubtree.second);

	return result;
}

__host__ std::pair< std::string, std::string > GraphToStringConverter::toString(
	Graph & aGraph,
	thrust::host_vector<unsigned int>& aNodeTypes)
{
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

	thrust::host_vector<unsigned int> visited(numNodes, 0u);
	thrust::host_vector<unsigned int> intervalsHost(aGraph.intervals);
	thrust::host_vector<unsigned int> adjacencyValsHost(aGraph.adjacencyVals);
	
	std::pair< std::string, std::string > result = depthFirstTraverse(
		0u,
		visited,
		(unsigned int)-1,
		intervalsHost,
		adjacencyValsHost,
		adjMatrixHost,
		cycleIds,
		aNodeTypes);

	result.second.erase(result.second.find_last_of(" "), 1); //remove last whitespace
	result.first.append("\n");
	result.second.append("\n");

	for (unsigned int startId = 1; startId < numNodes; ++startId)
	{
		visited = thrust::host_vector<unsigned int>(numNodes, 0u);
		std::pair< std::string, std::string > next = depthFirstTraverse(
			startId,
			visited,
			(unsigned int)-1,
			intervalsHost,
			adjacencyValsHost,
			adjMatrixHost,
			cycleIds,
			aNodeTypes);
		result.first.append(next.first);
		result.first.append("\n");

		next.second.erase(next.second.find_last_of(" "), 1);//remove last whitespace
		result.second.append(next.second);
		result.second.append("\n");
	}
	return result;
}

__host__ std::pair< std::string, std::string >
GraphToStringConverter::operator()(WFObject & aObj, Graph & aGraph)
{

	if (aObj.objects.size() != aGraph.numNodes())
	{
		std::cerr
			<< "Number of objects " << aObj.objects.size()
			<< " and graph nodes " << aGraph.numNodes()
			<< " do not match\n";
		return std::make_pair(std::string(""), std::string(""));
	}

	if (aObj.materials.size() > mAlphabet.size())
	{
		std::cerr << "Too many object types " << aObj.materials.size() << "\n";
		std::cerr << "Current maximum number is " << mAlphabet.size() << "\n";
		return std::make_pair(std::string(""), std::string(""));
	}

	thrust::host_vector<unsigned int> nodeTypes(aGraph.numNodes(), (unsigned int)aObj.materials.size());
	for (size_t nodeId = 0; nodeId < aObj.objects.size(); ++nodeId)
	{
		size_t faceId = aObj.objects[nodeId].x;
		size_t materialId = aObj.faces[faceId].material;
		nodeTypes[nodeId] = (unsigned int)materialId;
	}
	return toString(aGraph, nodeTypes); 

}

__host__ void GrammarCheck::init(
	thrust::host_vector<unsigned int>& aIntervals,
	thrust::host_vector<unsigned int>& aNbrIds,
	thrust::host_vector<unsigned int>& aNodeTypes)
{
	unsigned int newTypes = 1u + thrust::reduce(aNodeTypes.begin(), aNodeTypes.end(), 0u, thrust::maximum<unsigned int>());

	if (newTypes >= mNumTypes)
	{
		mNumTypes = newTypes;
		mNeighborCounts.resize(mNumTypes);
		mSupportFlags.resize(mNumTypes, true);
	}

	
	for (size_t i = 0; i < aIntervals.size() - 1; i++)
	{
		unsigned int typeId = aNodeTypes[i];
		unsigned int nbrCount = aIntervals[i + 1] - aIntervals[i];
		bool seenCount = false;
		for (size_t cntId = 0; cntId < mNeighborCounts[typeId].size() && !seenCount; ++cntId)
			if (nbrCount == mNeighborCounts[typeId][cntId])
				seenCount = true;
		if (!seenCount)
			mNeighborCounts[typeId].push_back(nbrCount);
		
		std::vector<unsigned int> nbrTypeCounts(mNumTypes, 0u);

		for (size_t nbrId = aIntervals[i]; nbrId < aIntervals[i+1]; nbrId++)
		{
			unsigned int nbrTypeId = aNodeTypes[aNbrIds[nbrId]];
			
			nbrTypeCounts[nbrTypeId]++;

			std::pair<unsigned int, unsigned int> nbrPair1 = std::make_pair(typeId, nbrTypeId);
			std::pair<unsigned int, unsigned int> nbrPair2 = std::make_pair(nbrTypeId, typeId);
			mNeighborTypes.insert(nbrPair1);
			mNeighborTypes.insert(nbrPair2);
		}

		mNeighborTypeCounts.insert(std::make_pair(typeId, nbrTypeCounts));
	}

}

__host__ void GrammarCheck::init(
	WFObject& aObj,
	thrust::device_vector<unsigned int>& aIntervals,
	thrust::device_vector<unsigned int>& aNbrIds)
{

	thrust::host_vector<unsigned int> intervals(aIntervals);
	thrust::host_vector<unsigned int> nbrIds(aNbrIds);
	thrust::host_vector<unsigned int> nodeTypes(aObj.objects.size(), (unsigned int)aObj.materials.size());
	for (size_t nodeId = 0; nodeId < aObj.objects.size(); ++nodeId)
	{
		size_t faceId = aObj.objects[nodeId].x;
		size_t materialId = aObj.faces[faceId].material;
		nodeTypes[nodeId] = (unsigned int)materialId;
	}

	unsigned int newTypes = 1u + thrust::reduce(nodeTypes.begin(), nodeTypes.end(), 0u, thrust::maximum<unsigned int>());

	if (newTypes >= mNumTypes)
	{
		mNumTypes = newTypes;
		mNeighborCounts.resize(mNumTypes);
		mSupportFlags.resize(mNumTypes, true);

	}
	thrust::host_vector<float3> objCenters;
	thrust::host_vector<float> objSizes;

	ObjectCenterExporter()(aObj, objCenters, objSizes);

	float3 minBound = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
	float3 maxBound = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	for (size_t i = 0; i < objCenters.size() - 1; i++)
	{
		minBound = min(minBound, objCenters[i]);
		maxBound = max(maxBound, objCenters[i]);
	}
	
	for (size_t i = 0; i < aIntervals.size() - 1; i++)
	{
		unsigned int typeId = nodeTypes[i];
		if (!mSupportFlags[typeId])
			continue;

		bool hasSupport = isOnTheGround(objCenters[i], minBound, maxBound);

		for (size_t nbrId = aIntervals[i]; nbrId < aIntervals[i + 1] && !hasSupport; nbrId++)
		{
			if (objCenters[i].z > objCenters[aNbrIds[nbrId]].z + 0.25f * objSizes[i])
				hasSupport = true;
		}
		if (!hasSupport)
			mSupportFlags[typeId] = false;
	}

	init(intervals, nbrIds, nodeTypes);
}

__host__ bool GrammarCheck::check(
	thrust::host_vector<unsigned int>& aIntervals,
	thrust::host_vector<unsigned int>& aNbrIds,
	thrust::host_vector<unsigned int>& aNodeTypes)
{
	unsigned int numTypes = 1u + thrust::reduce(aNodeTypes.begin(), aNodeTypes.end(), 0u, thrust::maximum<unsigned int>());
	if (numTypes != mNumTypes)
		return false;

	for (size_t i = 0; i < aIntervals.size() - 1; i++)
	{
		unsigned int typeId = aNodeTypes[i];
		unsigned int nbrCount = aIntervals[i + 1] - aIntervals[i];
		bool seenCount = false;
		for (size_t cntId = 0; cntId < mNeighborCounts[typeId].size() && !seenCount; ++cntId)
			if (nbrCount == mNeighborCounts[typeId][cntId])
				seenCount = true;
		if (!seenCount)
			return false;

		std::vector<unsigned int> nbrTypeCounts(mNumTypes, 0u);

		for (size_t nbrId = aIntervals[i]; nbrId < aIntervals[i + 1]; nbrId++)
		{
			unsigned int nbrTypeId = aNodeTypes[aNbrIds[nbrId]];
			nbrTypeCounts[nbrTypeId]++;

			std::pair<unsigned int, unsigned int> nbrPair1 = std::make_pair(typeId, nbrTypeId);
			if (mNeighborTypes.find(nbrPair1) == mNeighborTypes.end())
				return false;
		}

		std::pair<unsigned int, std::vector<unsigned int> > pair = std::make_pair(typeId, nbrTypeCounts);
		if (mNeighborTypeCounts.find(pair) == mNeighborTypeCounts.end())
			return false;
	}

	//check for diconnected components
	size_t numNodes = aIntervals.size() - 1;
	thrust::host_vector<unsigned int> visited(numNodes, 0u);

	std::deque<unsigned int> frontier;
	frontier.push_back(0);
	visited[0] = 1u;
	size_t visitedCount = 1u;
	while (!frontier.empty())
	{
		const unsigned int nodeId = frontier.front();
		frontier.pop_front();

		for (unsigned int nbrId = aIntervals[nodeId]; nbrId < aIntervals[nodeId + 1]; ++nbrId)
		{
			const unsigned int nodeId = aNbrIds[nbrId];
			if (visited[nodeId] == 0u)
			{
				frontier.push_back(nodeId);
				visited[nodeId] = 1u;
				++visitedCount;
			}
		}
	}

	if (visitedCount < numNodes)
		return false; // disconnected components


	return true;
}

__host__ bool GrammarCheck::checkSupport(
	WFObject& aObj,
	thrust::host_vector<unsigned int>& aIntervals,
	thrust::host_vector<unsigned int>& aNbrIds,
	thrust::host_vector<unsigned int>& aNodeTypes)
{

	thrust::host_vector<float3> objCenters;
	thrust::host_vector<float> objSizes;

	ObjectCenterExporter()(aObj, objCenters, objSizes);

	float3 minBound = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
	float3 maxBound = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	for (size_t i = 0; i < objCenters.size() - 1; i++)
	{
		minBound = min(minBound, objCenters[i]);
		maxBound = max(maxBound, objCenters[i]);
	}
	
	for (size_t i = 0; i < aIntervals.size() - 1; i++)
	{
		unsigned int typeId = aNodeTypes[i];
		if (!mSupportFlags[typeId])
			continue;

		bool hasSupport = isOnTheGround(objCenters[i], minBound, maxBound);

		for (size_t nbrId = aIntervals[i]; nbrId < aIntervals[i + 1] && !hasSupport; nbrId++)
		{
			if (objCenters[i].z > objCenters[aNbrIds[nbrId]].z + 0.25f * objSizes[i])
				hasSupport = true;
		}
		if (!hasSupport)
			return false;
	}
	return true;
}

__host__ void GrammarCheck::cleanup()
{
	mNumTypes = 0u;
	mNeighborCounts.clear();
	mNeighborTypes.clear();
	mNeighborTypeCounts.clear();
}
