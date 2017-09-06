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
#include "WFObjUtils.h"


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
		std::string result = converter(obj, graph).c_str();
		
		result = result.substr(0u, result.find_first_of("\n"));

		if (outputString != NULL)
			free(outputString);

		outputString = new char[result.length() + 1];
		strcpy(outputString, result.c_str());

		return outputString;
	}

	char * WFObjectToStrings(const char * aFilename)
	{
		WFObject obj;
		obj.read(aFilename);

		CollisionDetector detector;
		Graph graph = detector.computeCollisionGraph(obj, 0.0f);

		CollisionGraphExporter exporter;
		exporter.exportCollisionGraph(aFilename, obj, graph);


		GraphToStringConverter converter;
		std::string result = converter(obj, graph).c_str();

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

		for (size_t i = 0; i < 64u; ++i)
		{
			wiggle.fixRelativeTransformations(obj3, graph3);
			if (wiggle.numCorrections == 0u)
				break;
		}

		Graph modifiedGraph = detector.computeCollisionGraph(obj3, 0.0f);
		hostIntervals = thrust::host_vector<unsigned int>(modifiedGraph.intervals);
		hostNbrIds = thrust::host_vector<unsigned int>(modifiedGraph.adjacencyVals);

		if (grammarCheck.check(hostIntervals, hostNbrIds, nodeTypes))
		{
			//std::string fileName3(aFileName3);
			//if (fileName3.find_last_of("/\\") == std::string::npos)
			//	fileName3 = fileName3.substr(0, fileName3.size() - 5);
			//else
			//	fileName3 = fileName3.substr(fileName3.find_last_of("/\\") + 1, fileName3.size() - fileName3.find_last_of("/\\") - 5);

			//std::string objDir = getDirName(aFileName3);
			//std::string fixedFilePath = objDir + fileName3 + "_fixed";

			WFObjectFileExporter   objExporter;
			objExporter(obj3, aOutFileName);
		}
		else
		{
			//std::cerr << "Object repair attempt failed.\n";
			return 2;
		}

		return 0;

	}


	int testRandomNumberGenerator()
	{
		const unsigned int aConst1 = 101;
		const unsigned int aConst2 = 22;
		for (unsigned int tId = 0u; tId < 500u; ++tId)
		{
			KISSRandomNumberGenerator genRand(
				3643u + tId * 4154207u * aConst1 + aConst2,
				1761919u + tId * 2746753u * aConst1,
				331801u + tId,
				10499029u);

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
			}
		}


		return 0;

	}
#ifdef __cplusplus
}
#endif