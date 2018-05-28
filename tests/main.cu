#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#include "WFObjectToString.h"
#include "UniformGridSortBuilderTest.h"
#include "GraphTest.h"
#include "CollisionTest.h"
#include "ShapeVariationTest.h"
#include "RNGTest.h"
#include "WiggleTest.h"

#include <thrust/detail/config.h>

int main()
{

#if 0//THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n";
		return 1;
	}
#endif

	RNGTest rngTest;
	int rng_test_result = rngTest.testAll();
	//graphTest.testAll(1000);

	if (rng_test_result != 0)
	{
		std::cerr << "Random number generator test failed!" << std::endl;
		return rng_test_result;
	}
	else
	{
		std::cerr << "Random number generator test passed." << std::endl;
	}

	UniformGridSortBuildTest uniformGridTest;
    int ugrid_test_result = uniformGridTest.testAll("../scenes/church/church.obj", 32, 16, 24);
	
	if (ugrid_test_result != 0)
	{
		std::cerr << "Uniform grid construction test failed!" << std::endl;
		return ugrid_test_result;
	}
	else
	{
		std::cerr << "Uniform grid construction test passed." << std::endl;
	}
	
	GraphTest graphTest;
	int graph_test_result = graphTest.testAll(1000);
	//graphTest.testAll(1000);
	
	if (graph_test_result != 0)
	{
		std::cerr << "Graph construction test failed!" << std::endl;
		return graph_test_result;
	}
	else
	{
		std::cerr << "Graph construction test passed." << std::endl;
	}
	
	CollisionTest collTest;
	int coll_test_result = collTest.testAll("../scenes/castle/castle.obj");
	if (coll_test_result != 0)
	{
		std::cerr << "Collision detection test failed!" << std::endl;
		return coll_test_result;
	}
	else
	{
		std::cerr << "Collision detection test passed" << std::endl;
	}
	
	std::cerr << "---------------------------------------------------------------------\n";
	const char* obj2strTestFile = "../scenes/church/church.obj";
	std::cerr << obj2strTestFile << " converted to \n"
		<< WFObjectToStrings(obj2strTestFile) << std::endl;


	//const char* wiggleFile1 = "../scenes/wiggle_test/v001.obj";
	//const char* wiggleFile2 = "../scenes/wiggle_test/v002.obj";
	//const char* wiggleFile3 = "../scenes/wiggle_test/v001_v002_2_v002_2_v001_v002_3_v002_3_2_v001_v002_3_v002_3_v002_5_1.obj";
	//const char* wiggleFile3 = "../scenes/wiggle_test/v001_v001_v002_2_3_v001_v002_3_v002_3_3_v001_v002_2_v002_2_v001_v002_3_v002_3_3_3.obj";	
	//const char* wiggleFile3 = "../scenes/wiggle_test/v002.obj";

	const char* wiggleFile1 = "../scenes/wiggle_test/c19.obj";
	const char* wiggleFile2 = "../scenes/wiggle_test/c28.obj";
	//const char* wiggleFile3 = "../scenes/wiggle_test/v_1_4_11.obj";
	//const char* wiggleFile3 = "../scenes/wiggle_test/c19_c19_c28_12_1.obj";
	const char* wiggleFile3 = "../scenes/wiggle_test/v_1_4_12.obj";

	//const char* wiggleFile1 = "../scenes/skyscraper/v01.obj";
	//const char* wiggleFile2 = "../scenes/skyscraper/v02.obj";
	//const char* wiggleFile3 = "../scenes/skyscraper/v01_v02_5.obj";

	//const char* wiggleFile1 = "../scenes/sand_castle/v01.obj";
	//const char* wiggleFile2 = "../scenes/sand_castle/v02.obj";
	//const char* wiggleFile3 = "../scenes/wiggle_test/v01_v02_3.obj";

	//const char* wiggleFile1 = "../scenes/test_brick_house/b02.obj";
	//const char* wiggleFile2 = "../scenes/test_brick_house/b03.obj";
	//const char* wiggleFile3 = "../scenes/test_brick_house/b03_5.obj";


	const char* wiggleOutFile = "../scenes/wiggle_test/fixed.obj";

	WiggleTest wiggleTest;
	int wiggle_test_result = wiggleTest.testAll(wiggleFile1, wiggleFile2, wiggleFile3, wiggleOutFile);
	if (wiggle_test_result != 0)
	{
		std::cerr << "Wiggle test failed!" << std::endl;
		if (wiggle_test_result == 1)
		{
			std::cerr << "Invalid repair target - does not conform grammar." << std::endl;
		}
		else if (wiggle_test_result == 2)
		{
			std::cerr << "Object repair attempt failed." << std::endl;
		}
		//return wiggle_test_result;
	}
	else
	{
		std::cerr << "Object repair attempt succeeded.\n";
		std::cerr << "Wrote " << wiggleOutFile << std::endl;
	}


	std::cerr << "---------------------------------------------------------------------\n";
	const char* embeddingFile1 = "../scenes/test_sand_castle/v01.obj";
	const char* embeddingFile3 = "../scenes/test_sand_castle/embed01";
	const char* edgeVals = "48 42 39 38 13 5 12 32 31 14 7 19 18 3 11 8 9 23 30 29 28 44 36 35 33 17 27 49 26 10 41 35 1 25 43 21 24 45 20 46 4 34 2 15 47 22 50 16 37 6 40 7 0 48 42 39 38 13 5 12 32 31 14 0 19 18 3 11 8 9 23 30 29 28 44 36 23 33 17 27 49 26 23 41 23 1 25 43 21 24 23 20 46 4 34 2 15 47 22 50 16 37 6 40\n";
	const char* edgeKeys = "0 48 42 39 38 13 5 12 32 31 14 0 19 18 3 11 8 9 23 30 29 28 44 36 23 33 17 27 49 26 23 41 23 1 25 43 21 24 23 20 46 4 34 2 15 47 22 50 16 37 6 40 48 42 39 38 13 5 12 32 31 14 7 19 18 3 11 8 9 23 30 29 28 44 36 35 33 17 27 49 26 10 41 35 1 25 43 21 24 45 20 46 4 34 2 15 47 22 50 16 37 6 40 7\n";
	const char* edgeCats = "9 20 12 22 17 22 12 22 1 18 12 14 23 16 22 12 22 1 18 10 20 0 18 8 19 16 22 1 18 12 19 9 19 16 23 0 19 14 18 12 22 12 22 12 22 10 20 12 22 15 23 13 21 8 22 15 23 13 23 13 19 0 23 23 15 22 13 23 13 19 4 20 11 18 2 20 5 22 13 19 5 23 2 21 3 22 17 18 3 22 0 23 13 23 14 23 14 20 11 22 14 23 12 22\n";
	std::string graph1Str(edgeKeys);
	graph1Str.append(edgeVals);
	graph1Str.append(edgeCats);
	std::string completeSring = graph1Str + graph1Str + graph1Str;

	//const char* embeddingFile1 = "../scenes/test_playground/v001.obj";
	//const char* embeddingFile3 = "../scenes/test_playground/embed001";
	//const char* edgeCategoriesString1 = "1 47 34 35 40 14 2 46 61 4 22 20 66 32 7 33 60 58 55 11 46 10 6 61 57 64 16 31 53 64 36 45 29 56 23 30 44 63 62 28 13 15 38 51 67 3 52 65 9 39 54 8 37 59 21 70 68 48 42 27 41 24 49 43 25 19 12 69 18 26 25 22 5 17 50 0 1 47 34 34 34 14 2 2 61 61 2 20 66 66 66 2 60 58 55 11 11 10 6 10 10 11 16 31 53 53 36 45 45 45 53 31 31 55 55 28 13 15 38 38 38 38 52 65 65 65 13 13 13 59 21 21 21 1 42 27 41 41 41 42 42 1 12 69 69 26 26 26 69 1\n 0 1 47 34 34 34 14 2 2 61 61 2 20 66 66 66 2 60 58 55 11 11 10 6 10 10 11 16 31 53 53 36 45 45 45 53 31 31 55 55 28 13 15 38 38 38 38 52 65 65 65 13 13 13 59 21 21 21 1 42 27 41 41 41 42 42 1 12 69 69 26 26 26 69 1 1 47 34 35 40 14 2 46 61 4 22 20 66 32 7 33 60 58 55 11 46 10 6 61 57 64 16 31 53 64 36 45 29 56 23 30 44 63 62 28 13 15 38 51 67 3 52 65 9 39 54 8 37 59 21 70 68 48 42 27 41 24 49 43 25 19 12 69 18 26 25 22 5 17 50\n 29 0 10 6 7 0 15 4 31 7 1 0 10 7 6 8 6 2 18 31 3 31 3 16 7 3 3 19 31 4 3 16 8 7 6 7 6 8 6 1 13 1 13 9 7 6 0 11 7 8 6 7 6 0 11 7 8 6 34 3 16 7 9 6 0 7 0 10 9 31 1 0 7 6 6 8 12 1 21 23 11 1 16 33 23 10 13 1 23 21 26 22 5 3 32 17 32 17 4 23 18 18 4 32 19 17 4 30 23 21 23 21 26 21 11 0 14 0 28 24 22 12 1 23 30 21 24 22 12 1 23 30 21 33 20 4 23 30 21 14 24 14 1 27 32 11 10 23 21 22\n";
	//std::string completeSring(edgeCategoriesString1);
	//completeSring.append(edgeCategoriesString1);
	//completeSring.append(edgeCategoriesString1);

	int embedding_test_result = StringToWFObject(embeddingFile1, embeddingFile1, completeSring.c_str(), embeddingFile3);
	if (embedding_test_result != 0)
	{
		std::cerr << "String embedding test failed!" << std::endl;
		if (embedding_test_result == 1)
		{
			std::cerr << "Invalid embedding result - does not conform grammar." << std::endl;
		}

		//return embedding_test_result;
	}
	else
	{
		std::cerr << "String embedding attempt succeeded.\n";
		std::cerr << "Wrote " << embeddingFile3 << ".obj\n";
	}


	const char* variationFile1 = "../scenes/test_church/c19.obj";
	const char* variationFile2 = "../scenes/test_church/c28.obj";
	//const char* variationFile1 = "../scenes/test_skyscraper/v01.obj";
	//const char* variationFile2 = "../scenes/test_skyscraper/v02.obj";
	//const char* variationFile1 = "../scenes/test_sand_castle/v01.obj";
	//const char* variationFile2 = "../scenes/test_sand_castle/v02.obj";
	//const char* variationFile1 = "../scenes/test_playground/v001.obj";
	//const char* variationFile2 = "../scenes/test_playground/v002.obj";
	//const char* variationFile1 = "../scenes/moon_base/variant_01.obj";
	//const char* variationFile2 = "../scenes/moon_base/variant_02.obj";
	//const char* variationFile1 = "../scenes/test_brick_house/b01.obj";
	//const char* variationFile2 = "../scenes/test_brick_house/b02.obj";
	//const char* variationFile1 = "../scenes/test_brick_house/b02.obj";
	//const char* variationFile2 = "../scenes/test_brick_house/b03.obj";


	ShapeVariationTest variationTest;
	int var_test_result = variationTest.testAll(variationFile1, variationFile2);
	if (var_test_result != 0)
	{
		std::cerr << "Shape variation test failed!" << std::endl;
		return var_test_result;
	}
	else
	{
		std::cerr << "Shape variation test passed." << std::endl;
	}




#if 0//THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
     //cudaDeviceReset must be called before exiting in order for profiling and
     //tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
		std::cerr << "cudaDeviceReset failed!\n";
        return 1;
    }
#endif

    return 0;
}

