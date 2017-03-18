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

	char * WFObjectToString(const char * aFilename)
	{
		WFObject obj;
		obj.read(aFilename);

		CollisionDetector detector;
		Graph graph = detector.computeCollisionGraph(obj, 0.01f);

		GraphToStringConverter converter;
		std::string result = converter(obj, graph).c_str();

		char *cstr = new char[result.length() + 1];
		strcpy(cstr, result.c_str());

		return cstr;
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
		Graph testGraph = detector.computeCollisionGraph(testObj, 0.01f);
		detector.stats();

		CollisionGraphExporter exporter;
		exporter.exportCollisionGraph(aFilename, testObj, testGraph);
		exporter.stats();

		return 0;
	}


#ifdef __cplusplus
}
#endif