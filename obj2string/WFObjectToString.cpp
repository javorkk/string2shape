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
		UniformGrid grid = 	builder.build(testObj, aResX, aResY, aResZ);
		
		return builder.test(grid, testObj);
	}

	int testGraphConstruction(int aGraphSize)
	{
		std::vector<unsigned int> adjacencyMatrixHost(aGraphSize * aGraphSize);
		for(size_t i = 0; i < aGraphSize; ++i)
		{
			for (size_t j = 0; j < i; ++j)
			{
				bool makeEdge = rand() / RAND_MAX > 0.5f;
				if (makeEdge)
				{
					adjacencyMatrixHost[j * aGraphSize + i] = 1u;
					adjacencyMatrixHost[i * aGraphSize + j] = 1u;
				}
			}
		}
		thrust::device_vector<unsigned int> adjacencyMatrixDevice(adjacencyMatrixHost);
		Graph testGraph;
		testGraph.fromAdjacencyMatrix(adjacencyMatrixDevice, (size_t)aGraphSize);
		adjacencyMatrixDevice.clear();
		size_t newGrapSize;
		testGraph.toAdjacencyMatrix(adjacencyMatrixDevice, newGrapSize);
		for (size_t i = 0; i < aGraphSize * aGraphSize; ++i)
		{
			if (adjacencyMatrixDevice[i] != adjacencyMatrixHost[i])
			{
				std::cerr << "Wrong adjacency matrix value at position " << i;
				std::cerr << " device " << adjacencyMatrixDevice[i] << " ";
				std::cerr << " host " << adjacencyMatrixHost[i] << " ";
				std::cerr << "\n";
				return 1;
			}
		}


		return 0;
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