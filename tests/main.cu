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


	//const char* variationFile1 = "../scenes/test_church/c19.obj";
	//const char* variationFile2 = "../scenes/test_church/c28.obj";
	//const char* variationFile1 = "../scenes/test_skyscraper/v01.obj";
	//const char* variationFile2 = "../scenes/test_skyscraper/v02.obj";
	const char* variationFile1 = "../scenes/test_sand_castle/small.obj";
	const char* variationFile2 = "../scenes/test_sand_castle/v02.obj";
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

	std::cerr << "---------------------------------------------------------------------\n";
	const char* embeddingFile1 = "../scenes/test_sand_castle/v01.obj";
	const char* embeddingFile2 = "../scenes/test_sand_castle/v02.obj";
	const char* embeddingFile3 = "../scenes/test_sand_castle/embed01";
	const char* edgeVals = "48 42 39 38 13 5 12 32 31 14 7 19 18 3 11 8 9 23 30 29 28 44 36 35 33 17 27 49 26 10 41 35 1 25 43 21 24 45 20 46 4 34 2 15 47 22 50 16 37 6 40 7 0 48 42 39 38 13 5 12 32 31 14 0 19 18 3 11 8 9 23 30 29 28 44 36 23 33 17 27 49 26 23 41 23 1 25 43 21 24 23 20 46 4 34 2 15 47 22 50 16 37 6 40\n";
	const char* edgeKeys = "0 48 42 39 38 13 5 12 32 31 14 0 19 18 3 11 8 9 23 30 29 28 44 36 23 33 17 27 49 26 23 41 23 1 25 43 21 24 23 20 46 4 34 2 15 47 22 50 16 37 6 40 48 42 39 38 13 5 12 32 31 14 7 19 18 3 11 8 9 23 30 29 28 44 36 35 33 17 27 49 26 10 41 35 1 25 43 21 24 45 20 46 4 34 2 15 47 22 50 16 37 6 40 7\n";
	const char* edgeCats = "9 20 12 22 17 22 12 22 1 18 12 14 23 16 22 12 22 1 18 10 20 0 18 8 19 16 22 1 18 12 19 9 19 16 23 0 19 14 18 12 22 12 22 12 22 10 20 12 22 15 23 13 21 8 22 15 23 13 23 13 19 0 23 23 15 22 13 23 13 19 4 20 11 18 2 20 5 22 13 19 5 23 2 21 3 22 17 18 3 22 0 23 13 23 14 23 14 20 11 22 14 23 12 22\n";
	std::string graph1Str(edgeKeys);
	graph1Str.append(edgeVals);
	graph1Str.append(edgeCats);
	std::string completeSring = graph1Str + graph1Str + graph1Str;
	std::string  completeSring_2("48 42 39 38 13 5 12 32 31 14 7 19 18 3 11 8 9 23 20 46 4 34 2 15 47 22 50 16 37 6 40 7 41 30 29 28 44 36 35 41 1 25 43 21 24 45 33 17 27 49 26 10 0 48 42 39 38 13 5 12 32 31 14 0 19 18 3 11 8 9 23 20 46 4 34 2 15 47 22 50 16 37 6 40 23 23 30 29 28 44 36 35 23 1 25 43 21 24 23 33 17 27 49 26\n");
	completeSring_2.append("0 48 42 39 38 13 5 12 32 31 14 0 19 18 3 11 8 9 23 20 46 4 34 2 15 47 22 50 16 37 6 40 23 23 30 29 28 44 36 35 23 1 25 43 21 24 23 33 17 27 49 26 48 42 39 38 13 5 12 32 31 14 7 19 18 3 11 8 9 23 20 46 4 34 2 15 47 22 50 16 37 6 40 7 41 30 29 28 44 36 35 41 1 25 43 21 24 45 33 17 27 49 26 10\n");
	completeSring_2.append("6 18 10 20 15 20 10 20 0 16 10 12 21 14 20 10 20 0 16 10 20 10 20 10 20 9 18 10 20 13 21 11 17 16 9 18 1 16 7 19 17 14 21 1 17 12 17 14 20 0 16 10 19 7 20 13 21 11 21 11 17 1 21 21 13 20 11 21 11 17 1 21 11 21 12 21 12 18 8 20 12 21 10 20 2 3 18 8 16 2 18 6 4 20 15 16 4 20 5 20 11 17 5 21\n");
	completeSring_2.append("12 11 10 8 6 5 7 18 17 19 20 28 25 26 14 13 21 22 24 23 29 27 16 15 4 1 2 3 9 8 0 12 11 10 0 6 5 7 18 17 19 20 28 25 26 14 26 21 26 24 26 29 27 16 15 4 1 2 3 9\n");
	completeSring_2.append("0 12 11 10 0 6 5 7 18 17 19 20 28 25 26 14 26 21 26 24 26 29 27 16 15 4 1 2 3 9 12 11 10 8 6 5 7 18 17 19 20 28 25 26 14 13 21 22 24 23 29 27 16 15 4 1 2 3 9 8\n");
	completeSring_2.append("16 15 20 10 16 7 19 12 21 14 20 10 20 0 17 14 17 14 16 10 16 9 18 1 17 10 20 13 21 11 0 21 11 21 2 18 6 21 13 20 11 21 11 17 4 20 5 20 1 21 3 18 8 16 0 20 12 21 10 20\n");
	completeSring_2.append("1 0 2 1 3 2 4 3 5 4 6 5 7 6 8 7 9 8 10 9 11 10 12 11 13 12 14 13 15 14 16 15 17 16 18 17 19 18 20 19 21 20 22 21 23 22 24 23 25 24 26 25 27 26 28 0 29 28 30 29 31 30 32 31 33 32 34 33 35 34 36 35 37 36 38 37 39 38 40 39 41 40 27 41\n");
	completeSring_2.append("0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 10 11 11 12 12 13 13 14 14 15 15 16 16 17 17 18 18 19 19 20 20 21 21 22 22 23 23 24 24 25 25 26 26 27 0 28 28 29 29 30 30 31 31 32 32 33 33 34 34 35 35 36 36 37 37 38 38 39 39 40 40 41 41 27\n");
	completeSring_2.append("18 8 13 20 21 10 12 20 20 10 10 20 21 13 10 20 20 10 10 21 20 10 10 20 20 10 10 20 20 10 10 20 20 10 10 21 20 10 10 20 21 10 11 20 21 10 10 20 20 10 11 20 21 12 18 6 5 16 16 0 15 20 20 10 10 20 20 10 11 20 21 10 10 20 20 10 11 20 21 10 10 20 20 11\n");

	//const char* embeddingFile1 = "../scenes/test_playground/v001.obj";
	//const char* embeddingFile3 = "../scenes/test_playground/embed001";
	//const char* edgeCategoriesString1 = "1 47 34 35 40 14 2 46 61 4 22 20 66 32 7 33 60 58 55 11 46 10 6 61 57 64 16 31 53 64 36 45 29 56 23 30 44 63 62 28 13 15 38 51 67 3 52 65 9 39 54 8 37 59 21 70 68 48 42 27 41 24 49 43 25 19 12 69 18 26 25 22 5 17 50 0 1 47 34 34 34 14 2 2 61 61 2 20 66 66 66 2 60 58 55 11 11 10 6 10 10 11 16 31 53 53 36 45 45 45 53 31 31 55 55 28 13 15 38 38 38 38 52 65 65 65 13 13 13 59 21 21 21 1 42 27 41 41 41 42 42 1 12 69 69 26 26 26 69 1\n 0 1 47 34 34 34 14 2 2 61 61 2 20 66 66 66 2 60 58 55 11 11 10 6 10 10 11 16 31 53 53 36 45 45 45 53 31 31 55 55 28 13 15 38 38 38 38 52 65 65 65 13 13 13 59 21 21 21 1 42 27 41 41 41 42 42 1 12 69 69 26 26 26 69 1 1 47 34 35 40 14 2 46 61 4 22 20 66 32 7 33 60 58 55 11 46 10 6 61 57 64 16 31 53 64 36 45 29 56 23 30 44 63 62 28 13 15 38 51 67 3 52 65 9 39 54 8 37 59 21 70 68 48 42 27 41 24 49 43 25 19 12 69 18 26 25 22 5 17 50\n 29 0 10 6 7 0 15 4 31 7 1 0 10 7 6 8 6 2 18 31 3 31 3 16 7 3 3 19 31 4 3 16 8 7 6 7 6 8 6 1 13 1 13 9 7 6 0 11 7 8 6 7 6 0 11 7 8 6 34 3 16 7 9 6 0 7 0 10 9 31 1 0 7 6 6 8 12 1 21 23 11 1 16 33 23 10 13 1 23 21 26 22 5 3 32 17 32 17 4 23 18 18 4 32 19 17 4 30 23 21 23 21 26 21 11 0 14 0 28 24 22 12 1 23 30 21 24 22 12 1 23 30 21 33 20 4 23 30 21 14 24 14 1 27 32 11 10 23 21 22\n";
	//std::string completeSring(edgeCategoriesString1);
	//completeSring.append(edgeCategoriesString1);
	//completeSring.append(edgeCategoriesString1);

	//const char* embeddingFile1 = "../scenes/test_church/c19.obj";
	//const char* embeddingFile2 = "../scenes/test_church/c28.obj";
	//const char* embeddingFile3 = "../scenes/test_church/embed01";
	//std::string  completeSring_2("17 0 29 17 30 29 33 30 39 29 41 39 26 41 20 26 25 20 2 25 24 2 14 24 28 14 16 28 9 16 23 9 36 23 22 36 15 16 31 15 4 31 19 4 27 19 18 27 35 18 1 27 12 1 38 12 42 38 5 42 40 5 43 40 34 43 8 34 25 8 6 5 10 0 7 10 32 7 37 32 3 37 11 3 6 11 13 32 21 13\n");
	//completeSring_2.append("(0 17 17 29 29 30 30 33 29 39 39 41 41 26 26 20 20 25 25 2 2 24 24 14 14 28 28 16 16 9 9 23 23 36 36 22 16 15 15 31 31 4 4 19 19 27 27 18 18 35 27 1 1 12 12 38 38 42 42 5 5 40 40 43 43 34 34 8 8 25 5 6 0 10 10 7 7 32 32 37 37 3 3 11 11 6 32 13 13 21\n");
	//completeSring_2.append("0 1 7 2 3 8 10 4 2 9 1 0 1 0 1 0 7 3 2 8 1 0 1 0 1 0 9 3 2 7 1 0 0 0 11 6 2 8 0 0 0 1 0 1 7 2 3 8 10 4 3 9 0 1 0 1 0 1 7 2 2 9 1 0 1 0 0 0 9 2 3 8 1 0 0 0 8 2 2 9 1 0 1 0 1 0 3 7 11 6\n");
	//completeSring_2.append("30 0 49 30 28 0 11 28 10 11 60 10 19 60 46 19 17 19 68 17 8 68 9 8 20 9 39 20 66 39 62 66 42 62 44 42 27 44 69 27 35 44 3 35 12 20 43 12 61 0 41 61 50 41 23 50 63 23 53 63 51 53 58 63 5 58 29 5 4 29 25 4 22 25 1 22 47 1 52 47 54 52 67 54 38 67 14 38 31 14 16 31 34 16 57 34 13 57 24 13 2 24 36 2 15 36 26 2 48 26 32 16 7 32 18 54 55 18 56 55 33 56 6 33 21 6 65 21 40 65 46 40 37 6 59 37 51 59 45 25 64 45\n");
	//completeSring_2.append("0 30 30 49 0 28 28 11 11 10 10 60 60 19 19 46 19 17 17 68 68 8 8 9 9 20 20 39 39 66 66 62 62 42 42 44 44 27 27 69 44 35 35 3 20 12 12 43 0 61 61 41 41 50 50 23 23 63 63 53 53 51 63 58 58 5 5 29 29 4 4 25 25 22 22 1 1 47 47 52 52 54 54 67 67 38 38 14 14 31 31 16 16 34 34 57 57 13 13 24 24 2 2 36 36 15 2 26 26 48 16 32 32 7 54 18 18 55 55 56 56 33 33 6 6 21 21 65 65 40 40 46 6 37 37 59 59 51 25 45 45 64\n");
	//completeSring_2.append("2 7 11 6 2 8 0 0 0 1 0 1 7 2 3 9 3 8 0 1 0 1 0 1 8 2 2 9 1 0 1 0 1 0 9 3 3 8 10 4 2 7 10 5 3 7 11 6 3 9 0 1 0 1 0 1 8 2 2 9 1 0 3 7 0 1 0 1 0 1 9 2 2 7 1 0 1 0 0 0 8 2 3 7 0 1 0 1 0 1 7 2 3 9 0 1 0 1 0 1 9 2 3 8 10 4 3 7 11 6 3 8 10 4 2 9 1 0 1 0 1 0 8 3 2 7 1 0 1 0 1 0 2 9 0 0 0 1 3 8 10 4\n");
	//completeSring_2.append("1 0 2 1 3 2 4 3 5 2 6 5\n");
	//completeSring_2.append("0 1 1 2 2 3 3 4 2 5 5 6\n");
	//completeSring_2.append("4 11 9 2 2 10 12 6 2 14 12 6\n");



	int embedding_test_result = StringToWFObject(embeddingFile1, embeddingFile2, completeSring.c_str(), embeddingFile3);
	if (embedding_test_result != 0)
	{
		std::cerr << "String embedding test failed!" << std::endl;
		if (embedding_test_result == 1)
		{
			std::cerr << "Non-strict embedding attempt succeeded.\n";
			std::cerr << "Wrote " << embeddingFile3 << ".obj\n";
		}
		else if (embedding_test_result == 2)
		{
			std::cerr << "Invalid embedding result - does not conform grammar." << std::endl;
			std::cerr << "Wrote largest valid subgraph in " << embeddingFile3 << "_best_subgraph.obj\n";
		}

		//return embedding_test_result;
	}
	else
	{
		std::cerr << "String embedding attempt succeeded.\n";
		std::cerr << "Wrote " << embeddingFile3 << ".obj\n";
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

