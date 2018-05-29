#include "pch.h"
#include "WFObjectToString.h"

#include "WFObject.h"
#include "UniformGrid.h"
#include "UniformGridSortBuilder.h"

#include "Graph.h"
#include "CollisionDetector.h"
#include "CollisionGraphExporter.h"
#include "Graph2String.h"
#include "VariationGenerator.h"
#include "Wiggle.h"
#include "PartOrientationUtils.h"
#include "WFObjUtils.h"
#include "RNG.h"
#include "GraphToWFObject.h"

#include "DebugUtils.h"


#ifdef __cplusplus
extern "C" {
#endif
	char * outputString = NULL;

	char * WFObjectToString(const char * aFilename)
	{
		WFObject obj;
		obj.read(aFilename);

		CollisionDetector detector;
		Graph graph = detector.computeCollisionGraph(obj, 0.0f);

		CollisionGraphExporter exporter;
		exporter.exportCollisionGraph(aFilename, obj, graph);


		GraphToStringConverter converter;
		std::string result = converter(obj, graph).first.c_str();
		
		result = result.substr(0u, result.find_first_of("\n"));

		if (outputString != NULL)
			free(outputString);

		outputString = new char[result.length() + 1];
		strcpy(outputString, result.c_str());

		return outputString;
	}

	char * WFObjectToStrings(const char * aFilename, bool aAppendNodeIds/* = false*/)
	{
		WFObject obj;
		obj.read(aFilename);

		CollisionDetector detector;
		Graph graph = detector.computeCollisionGraph(obj, 0.0f);

		CollisionGraphExporter exporter;
		exporter.exportCollisionGraph(aFilename, obj, graph);

		if (!graph.isConnected())
			return "";

		GraphToStringConverter converter;
		std::pair< std::string, std::string > strings_nodeIds = converter(obj, graph);

		for (size_t i = 0; i < 10; ++i)
		{
			std::pair< std::string, std::string > new_sample = converter(obj, graph);
			strings_nodeIds.first.append(new_sample.first);
			strings_nodeIds.second.append(new_sample.second);
		}

		std::string result = strings_nodeIds.first;

		if (aAppendNodeIds)
			result.append(strings_nodeIds.second.substr(0u, strings_nodeIds.second.find_last_of("\n")));

		if (outputString != NULL)
			free(outputString);

		outputString = new char[result.length() + 1];
		strcpy(outputString, result.c_str());

		return outputString;
	}

	char * WFObjectRandomVariations(const char * aFileName1, const char* aFileName2)
	{
		WFObject obj1;
		obj1.read(aFileName1);

		WFObject obj2;
		obj2.read(aFileName2);

		CollisionDetector detector;
		Graph graph1 = detector.computeCollisionGraph(obj1, 0.0f);
		Graph graph2 = detector.computeCollisionGraph(obj2, 0.0f);

		VariationGenerator genRandVariation;
		genRandVariation.writeVariationGraphs = false;
		genRandVariation.writeVariations = true;
		//genRandVariation.fixVariation = true;

		std::string result = genRandVariation(aFileName1, aFileName2, obj1, obj2, graph1, graph2, 0.0f);

		outputString = new char[result.length() + 1];
		strcpy(outputString, result.c_str());

		return outputString;
	}

	int buildGrid(const char * aFilename, int aResX, int aResY, int aResZ)
	{
		WFObject testObj;
		testObj.read(aFilename);

		UniformGridSortBuilder builder;
		UniformGrid grid = builder.build(testObj, aResX, aResY, aResZ);
		builder.stats();

		int result = builder.test(grid, testObj);
		grid.cleanup();

		return result;
	}

	int testGraphConstruction(int aGraphSize)
	{
		Graph testGraph;
		return testGraph.testGraphConstruction(aGraphSize);
	}

	int testCollisionGraphConstruction(const char * aFilename)
	{
		WFObject testObj;
		testObj.read(aFilename);

		CollisionDetector detector;
		Graph testGraph = detector.computeCollisionGraph(testObj, 0.02f);
		detector.stats();		

		CollisionGraphExporter exporter;
		exporter.exportCollisionGraph(aFilename, testObj, testGraph);
		exporter.stats();

		return testGraph.testSpanningTreeConstruction();
	}

	int testRandomVariations(const char * aFileName1, const char* aFileName2)
	{
		WFObject obj1;
		obj1.read(aFileName1);

		WFObject obj2;
		obj2.read(aFileName2);

		CollisionDetector detector;
		Graph graph1 = detector.computeCollisionGraph(obj1, 0.0f);
		Graph graph2 = detector.computeCollisionGraph(obj2, 0.0f);

		CollisionGraphExporter exporter;
		exporter.exportCollisionGraph(aFileName1, obj1, graph1);
		exporter.stats();

		exporter.exportCollisionGraph(aFileName2, obj2, graph2);
		exporter.stats();

		VariationGenerator genRandVariation;
		genRandVariation.writeVariations = true;
		//genRandVariation.fixVariation = true;
		genRandVariation(aFileName1, aFileName2, obj1, obj2, graph1, graph2, 0.0f);
		genRandVariation.stats();

		return 0;

	}

	int fixVariation(const char * aFileName1, const char* aFileName2, const char* aFileName3, const char* aOutFileName)
	{
		WFObject obj1;
		obj1.read(aFileName1);

		WFObject obj2;
		obj2.read(aFileName2);

		WFObject obj3;
		obj3.read(aFileName3);

		CollisionDetector detector;
		Graph graph1 = detector.computeCollisionGraph(obj1, 0.0f);
		Graph graph2 = detector.computeCollisionGraph(obj2, 0.0f);
		Graph graph3 = detector.computeCollisionGraph(obj3, 0.0f);

		GrammarCheck grammarCheck;
		grammarCheck.init(obj1, graph1.intervals, graph1.adjacencyVals);
		grammarCheck.init(obj2, graph2.intervals, graph2.adjacencyVals);

		thrust::host_vector<unsigned int> nodeTypes(graph3.numNodes());
		for (size_t nodeId = 0; nodeId < graph3.numNodes(); ++nodeId)
		{
			size_t faceId = obj3.objects[nodeId].x;
			size_t materialId = obj3.faces[faceId].material;
			nodeTypes[nodeId] = (unsigned int)materialId;
		}
		thrust::host_vector<unsigned int> hostIntervals(graph3.intervals);
		thrust::host_vector<unsigned int> hostNbrIds(graph3.adjacencyVals);
		if (!grammarCheck.check(hostIntervals, hostNbrIds, nodeTypes))
		{
			//std::cerr << "Invalid repair target - does not conform grammar.\n";
			return 1;
		}


		Wiggle wiggle;
		wiggle.init(obj1, graph1);
		wiggle.init(obj2, graph2);
		//wiggle.debugOutputLocalFrames = true;

		for (size_t i = 0; i < 128u; ++i)
		{
			wiggle.fixRelativeTransformations(obj3, graph3);
			if (wiggle.numCorrections == 0u)
				break;
		}

		Graph modifiedGraph = detector.computeCollisionGraph(obj3, 0.0f);
		hostIntervals = thrust::host_vector<unsigned int>(modifiedGraph.intervals);
		hostNbrIds = thrust::host_vector<unsigned int>(modifiedGraph.adjacencyVals);

		if (wiggle.debugOutputLocalFrames || grammarCheck.check(hostIntervals, hostNbrIds, nodeTypes))
		{
			std::string outFileName(aOutFileName);
			if (outFileName.find_last_of("/\\") == std::string::npos)
				outFileName = outFileName.substr(0, outFileName.size() - 5);
			else
				outFileName = outFileName.substr(outFileName.find_last_of("/\\") + 1, outFileName.size() - outFileName.find_last_of("/\\") - 5);

			std::string objDir = getDirName(aFileName3);
			std::string fixedFilePath = objDir + outFileName;

			WFObjectFileExporter   objExporter;
			objExporter(obj3, fixedFilePath.c_str());
		}
		else
		{
			//std::cerr << "Object repair attempt failed.\n";
			return 2;
		}

		return 0;

	}

	int StringToWFObject(
		const char * aFileName1,
		const char * aFileName2,
		const char * aInputString,
		const char * aOutFileName)
	{
		WFObject obj1;
		obj1.read(aFileName1);

		WFObject obj2;
		obj2.read(aFileName2);

		CollisionDetector detector;
		Graph graph1 = detector.computeCollisionGraph(obj1, 0.0f);
		Graph graph2 = detector.computeCollisionGraph(obj2, 0.0f);

		std::stringstream ss(aInputString);
		std::string NodesStr_1_1;
		std::getline(ss, NodesStr_1_1, '\n');
		std::string NodesStr_1_2;
		std::getline(ss, NodesStr_1_2, '\n');
		std::string EdgeTypeStr_1;
		std::getline(ss, EdgeTypeStr_1, '\n');

		thrust::host_vector<unsigned int> keys1(graph1.adjacencyKeys);
		thrust::host_vector<unsigned int> vals1(graph1.adjacencyVals);
		thrust::host_vector<unsigned int> edgeTypes1(graph1.numEdges() * 2, (unsigned int)-1);

		std::stringstream nodes_ss_1_1(NodesStr_1_1);
		std::stringstream nodes_ss_1_2(NodesStr_1_2);
		std::stringstream edge_types_1(EdgeTypeStr_1);
		unsigned int node_1, node_2, edge_type;
		while (nodes_ss_1_1 >> node_1 && nodes_ss_1_2 >> node_2 && edge_types_1 >> edge_type)
		{
			for (unsigned int edgeId = 0u; edgeId < graph1.numEdges() * 2; ++edgeId)
			{
				if (keys1[edgeId] == node_1 && vals1[edgeId] == node_2)
				{
					edgeTypes1[edgeId] = edge_type;
					break;
				}
			}
		}

		//outputHostVector("keys1     : ", keys1);
		//outputHostVector("vals1     : ", vals1);
		//outputHostVector("edgeTypes1: ", edgeTypes1);

		std::string NodesStr_2_1;
		std::getline(ss, NodesStr_2_1, '\n');
		std::string NodesStr_2_2;
		std::getline(ss, NodesStr_2_2, '\n');
		std::string EdgeTypeStr_2;
		std::getline(ss, EdgeTypeStr_2, '\n');

		thrust::host_vector<unsigned int> keys2(graph2.adjacencyKeys);
		thrust::host_vector<unsigned int> vals2(graph2.adjacencyVals);
		thrust::host_vector<unsigned int> edgeTypes2(graph2.numEdges() * 2, (unsigned int)-1);
		std::stringstream nodes_ss_2_1(NodesStr_2_1);
		std::stringstream nodes_ss_2_2(NodesStr_2_2);
		std::stringstream edge_types_2(EdgeTypeStr_2);
		while (nodes_ss_2_1 >> node_1 && nodes_ss_2_2 >> node_2 && edge_types_2 >> edge_type)
		{
			for (unsigned int edgeId = 0u; edgeId < graph2.numEdges() * 2; ++edgeId)
			{
				if (keys2[edgeId] == node_1 && vals2[edgeId] == node_2)
				{
					edgeTypes2[edgeId] = edge_type;
					break;
				}
			}
		}

		//outputHostVector("keys2     : ", keys2);
		//outputHostVector("vals2     : ", vals2);
		//outputHostVector("edgeTypes2: ", edgeTypes2);

		std::string NodesStr_3_1;
		std::getline(ss, NodesStr_3_1, '\n');
		std::string NodesStr_3_2;
		std::getline(ss, NodesStr_3_2, '\n');
		std::string EdgeTypeStr_3;
		std::getline(ss, EdgeTypeStr_3, '\n');

		std::vector<unsigned int> keys3;
		std::vector<unsigned int> vals3;
		std::vector<unsigned int> edgeTypes3;
		std::stringstream nodes_ss_3_1(NodesStr_3_1);
		std::stringstream nodes_ss_3_2(NodesStr_3_2);
		std::stringstream edge_types_3(EdgeTypeStr_3);
		unsigned int max_node_id = 0u;
		while (nodes_ss_3_1 >> node_1 && nodes_ss_3_2 >> node_2 && edge_types_3 >> edge_type)
		{
			max_node_id = std::max(max_node_id, node_1);
			max_node_id = std::max(max_node_id, node_2);
			keys3.push_back(node_1);
			vals3.push_back(node_2);
			edgeTypes3.push_back(edge_type);
		}
		
		Graph graph3;
		graph3.adjacencyKeys = thrust::device_vector<unsigned int>(keys3);
		graph3.adjacencyVals = thrust::device_vector<unsigned int>(vals3);
		graph3.fromAdjacencyList(max_node_id + 1, false);

		thrust::host_vector<unsigned int> edgeTypes3Host(graph3.numEdges() * 2, (unsigned)-1);
		for (unsigned int i = 0u; i < edgeTypes3.size(); ++i)
		{
			unsigned int nodeId_1 = keys3[i];
			unsigned int nodeId_2 = vals3[i];
			unsigned int edgeTypeId = edgeTypes3[i];
			unsigned int edgeId = graph3.getEdgeId(nodeId_1, nodeId_2);
			edgeTypes3Host[edgeId] = edgeTypeId;
		}

		//outputDeviceVector("keys3     : ", graph3.adjacencyKeys);
		//outputDeviceVector("vals3     : ", graph3.adjacencyVals);
		//outputHostVector("edgeTypes3: ", edgeTypes3Host);

		WFObjectGenerator embedGraphAsObj;

		///////////////////////////
		//check edge configurations
		///////////////////////////
		bool allEdgeTypesFound = true;
		for (unsigned int edgeId3 = 0; edgeId3 < edgeTypes3Host.size(); ++edgeId3)
		{
			unsigned int edgeTypeId3 = edgeTypes3Host[edgeId3];
			unsigned int node3A = graph3.adjacencyKeys[edgeId3];
			unsigned int node3B = graph3.adjacencyVals[edgeId3];

			unsigned int oposingEdgeId3 = graph3.getOpositeEdgeId(edgeId3);
			unsigned int oposingTypeId3 = edgeTypes3Host[oposingEdgeId3];
			
			bool foundIn1 = false;
			for (unsigned int edgeId1 = 0; edgeId1 < edgeTypes1.size(); ++edgeId1)
			{
				unsigned int edgeTypeId1 = edgeTypes1[edgeId1];

				unsigned int oposingEdgeId1 = graph1.getOpositeEdgeId(edgeId1);
				unsigned int oposingTypeId1 = edgeTypes1[oposingEdgeId1];

				if (edgeTypeId3 == edgeTypeId1 && oposingTypeId3 == oposingTypeId1)
				{
					foundIn1 = true;
					break;
				}
			}
			bool foundIn2 = false;
			for (unsigned int edgeId2 = 0; edgeId2 < edgeTypes2.size(); ++edgeId2)
			{
				unsigned int edgeTypeId2 = edgeTypes2[edgeId2];

				unsigned int oposingEdgeId2 = graph2.getOpositeEdgeId(edgeId2);
				unsigned int oposingTypeId2 = edgeTypes2[oposingEdgeId2];

				if (edgeTypeId3 == edgeTypeId2 && oposingTypeId3 == oposingTypeId2)
				{
					foundIn2 = true;
					break;
				}
			}
			if (!foundIn1 && !foundIn2)
			{
				allEdgeTypesFound = false;
				//std::cerr << "Did not find type pair " << edgeTypeId3 << " and " << oposingTypeId3 << "\n";
				//std::cerr << "Node ids in target graph " << node3A << " and " << node3B << "\n";							
			}
		}

		if(!allEdgeTypesFound)
			std::cerr << "Did not find examples for some of the requested edge type pairs in the input shapes.\n";

		GrammarCheck grammarCheck;
		grammarCheck.init(obj1, graph1.intervals, graph1.adjacencyVals);
		grammarCheck.init(obj2, graph2.intervals, graph2.adjacencyVals);

		for (size_t attempt = 0; attempt < 16; ++attempt)
		{
			WFObject obj3 = embedGraphAsObj(obj1, obj2, graph1, graph2, graph3, edgeTypes1, edgeTypes2, edgeTypes3Host);
			
			if (obj3.getNumObjects() <= 0)
				continue;

			Graph graph3_1 = detector.computeCollisionGraph(obj3, 0.0f);
			thrust::host_vector<unsigned int> nodeTypes(graph3_1.numNodes());
			for (size_t nodeId = 0; nodeId < graph3_1.numNodes(); ++nodeId)
			{
				size_t faceId = obj3.objects[nodeId].x;
				size_t materialId = obj3.faces[faceId].material;
				nodeTypes[nodeId] = (unsigned int)materialId;
			}
			thrust::host_vector<unsigned int> hostIntervals(graph3_1.intervals);
			thrust::host_vector<unsigned int> hostNbrIds(graph3_1.adjacencyVals);

			if (grammarCheck.check(hostIntervals, hostNbrIds, nodeTypes))
			{
				WFObjectFileExporter()(obj3, aOutFileName);
				if(obj3.getNumObjects() == graph3.numNodes())
					return 0;
			}
		}

		embedGraphAsObj.strictEmbeddingFlag = false;
		for (size_t attempt = 0; attempt < 16; ++attempt)
		{
			WFObject obj3 = embedGraphAsObj(obj1, obj2, graph1, graph2, graph3, edgeTypes1, edgeTypes2, edgeTypes3Host);
			
			if (obj3.getNumObjects() <= 0)
				continue;

			Graph graph3_1= detector.computeCollisionGraph(obj3, 0.0f);
			thrust::host_vector<unsigned int> nodeTypes(graph3_1.numNodes());
			for (size_t nodeId = 0; nodeId < graph3_1.numNodes(); ++nodeId)
			{
				size_t faceId = obj3.objects[nodeId].x;
				size_t materialId = obj3.faces[faceId].material;
				nodeTypes[nodeId] = (unsigned int)materialId;
			}
			thrust::host_vector<unsigned int> hostIntervals(graph3_1.intervals);
			thrust::host_vector<unsigned int> hostNbrIds(graph3_1.adjacencyVals);

			if (grammarCheck.check(hostIntervals, hostNbrIds, nodeTypes))
			{
				std::cerr << "Found a valid graph embedding using not strictly matching edge category pairs.\n";
				WFObjectFileExporter()(obj3, aOutFileName);
				return 0;
			}
		}

		std::string lastAttemptObjFileName = std::string(aOutFileName) + std::string("_attempt");
		
		//embedGraphAsObj.strictEmbeddingFlag = true;
		WFObject obj3 = embedGraphAsObj(obj1, obj2, graph1, graph2, graph3, edgeTypes1, edgeTypes2, edgeTypes3Host);

		//Wiggle wiggle;
		//wiggle.init(obj1, graph1);
		//wiggle.init(obj2, graph2);
		////wiggle.debugOutputLocalFrames = true;

		//for (size_t i = 0; i < 128u; ++i)
		//{
		//	wiggle.fixRelativeTransformations(obj3, graph3);
		//	if (wiggle.numCorrections == 0u)
		//		break;
		//}

		WFObjectFileExporter()(obj3, lastAttemptObjFileName.c_str());

		//std::cerr << "Failed to find a valid shape embedding. Writing an attempt in " << lastAttemptObjFileName << ".obj\n";

		return 1;
	}


	int testRandomNumberGenerator()
	{
		const unsigned int aConst1 = (unsigned int)std::chrono::system_clock::now().time_since_epoch().count();
		const unsigned int aConst2 = 22;
		for (unsigned int tId = 0u; tId < 500u; ++tId)
		{
			//KISSRandomNumberGenerator genRand(
			//	3643u + aConst1 + aConst2 * aConst2,
			//	aConst1,
			//	331801u + aConst2 * aConst1,
			//	10499029u);

			XorShift32Plus genRand(aConst1 + aConst2, aConst2 * aConst2 * aConst1);

			bool between0000_0125 = false;
			bool between0125_0025 = false;
			bool between0250_0375 = false;
			bool between0375_0500 = false;
			bool between0500_0625 = false;
			bool between0625_0750 = false;
			bool between0750_0875 = false;
			bool between0875_1000 = false;
			for (unsigned int testId = 0u; testId < 10000u; ++testId)
			{
				const float r = genRand();
				if (r < -EPS)
				{
					std::cerr << "Random number " << r << " < 0\n";
					return 1;
				}

				if (r > 1.f + EPS)
				{
					std::cerr << "Random number " << r << " > 1\n";
					return 2;
				}

				if (r < 0.125f)
				{
					between0000_0125 = true;
				}
				else if (r < 0.25f)
				{
					between0125_0025 = true;
				}
				else if (r < 0.375f)
				{
					between0250_0375 = true;
				}
				else if (r < 0.5f)
				{
					between0375_0500 = true;
				}
				else if (r < 0.625f)
				{
					between0500_0625 = true;
				}
				else if (r < 0.75f)
				{
					between0625_0750 = true;
				}
				else if (r < 0.875f)
				{
					between0750_0875 = true;
				}
				else
				{
					between0875_1000 = true;
				}
			}

			if (!(between0000_0125
				&& between0125_0025
				&& between0250_0375
				&& between0375_0500
				&& between0500_0625
				&& between0625_0750
				&& between0750_0875
				&& between0875_1000))
			{
				std::cerr << "Failed to cover each of eight bins [0,1] with 10000 samples.\n";
				return 3;
			}
		}


		return 0;

	}
#ifdef __cplusplus
}
#endif

std::vector<float> WFObjectToGraph(const char * aFilename)
{
	WFObject obj;
	obj.read(aFilename);

	CollisionDetector detector;
	Graph graph = detector.computeCollisionGraph(obj, 0.0f);

	PartOrientationEstimator orientationEstimator;
	orientationEstimator.init(obj, graph);

	return orientationEstimator.getEdgesTypesAndOrientations();
}
