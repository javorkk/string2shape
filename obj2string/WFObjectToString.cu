#include "pch.h"
#include "WFObjectToString.h"

#include "WFObject.h"
#include "UniformGrid.h"
#include "UniformGridSortBuilder.h"

#include "Graph.h"
#include "CollisionDetector.h"
#include "CollisionGraphExporter.h"


#ifdef __cplusplus
extern "C" {
#endif

	char * WFObjectToString(char * aFilename)
	{
		char* testFileName = "scenes/castle.obj";
		WFObject testObj;

		testObj.read(testFileName);

		UniformGridSortBuilder builder;
		UniformGrid grid = builder.build(testObj, 24, 24, 24);

		return testFileName;
	}

	int buildGrid(const char * aFilename, int aResX, int aResY, int aResZ)
	{
		WFObject testObj;
		testObj.read(aFilename);

		UniformGridSortBuilder builder;
		UniformGrid grid = builder.build(testObj, aResX, aResY, aResZ);

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
		Graph testGraph = detector.computeCollisionGraph(testObj, 0.05f);

		CollisionGraphExporter exporter;
		exporter.exportCollisionGraph("collision_graph", testObj, testGraph);

		return 0;
	}


#ifdef __cplusplus
}
#endif