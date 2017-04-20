#include "pch.h"
#include "WFObjectToString.h"

#include "WFObject.h"
#include "UniformGrid.h"
#include "UniformGridSortBuilder.h"

#include "Graph.h"
#include "CollisionDetector.h"
#include "CollisionGraphExporter.h"
#include "Graph2String.h"


#ifdef __cplusplus
extern "C" {
#endif
	char * outputString = NULL;

	char * WFObjectToString(const char * aFilename)
	{
		WFObject obj;
		obj.read(aFilename);

		CollisionDetector detector;
		Graph graph = detector.computeCollisionGraph(obj, 0.02f);

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
		Graph graph = detector.computeCollisionGraph(obj, 0.02f);

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

	int buildGrid(const char * aFilename, int aResX, int aResY, int aResZ)
	{
		WFObject testObj;
		testObj.read(aFilename);

		UniformGridSortBuilder builder;
		UniformGrid grid = builder.build(testObj, aResX, aResY, aResZ);
		builder.stats();

		return builder.test(grid, testObj);
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


#ifdef __cplusplus
}
#endif