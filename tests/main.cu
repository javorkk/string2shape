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
		std::cerr << "Random number generator test failed!\n";
		return rng_test_result;
	}
	else
	{
		std::cerr << "Random number generator test passed.\n";
	}

	UniformGridSortBuildTest uniformGridTest;
	int ugrid_test_result = uniformGridTest.testAll("../scenes/church/church.obj", 32, 16, 24);
	
	if (ugrid_test_result != 0)
	{
		std::cerr << "Uniform grid construction test failed!\n";
		return ugrid_test_result;
	}
	else
	{
		std::cerr << "Uniform grid construction test passed.\n";

	}
	
	GraphTest graphTest;
	int graph_test_result = graphTest.testAll(1000);
	//graphTest.testAll(1000);
	
	if (graph_test_result != 0)
	{
		std::cerr << "Graph construction test failed!\n";
		return graph_test_result;
	}
	else
	{
		std::cerr << "Graph construction test passed.\n";
	}
	
	CollisionTest collTest;
	int coll_test_result = collTest.testAll("../scenes/castle/castle.obj");
	if (coll_test_result != 0)
	{
		std::cerr << "Collision detection test failed!\n";
		return coll_test_result;
	}
	else
	{
		std::cerr << "Collision detection test passed. \n";
	}
	
	std::cerr << "---------------------------------------------------------------------\n";
	const char* obj2strTestFile = "../scenes/church/church.obj";
	std::cerr << obj2strTestFile << " converted to \n"
		<< WFObjectToString(obj2strTestFile) << "\n";


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

	ShapeVariationTest variationTest;
	int var_test_result = variationTest.testAll(variationFile1, variationFile2);
	if (var_test_result != 0)
	{
		std::cerr << "Shape variation test failed!\n";
		return var_test_result;
	}
	else
	{
		std::cerr << "Shape variation test passed. \n";
	}


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


	const char* wiggleOutFile = "../scenes/wiggle_test/fixed.obj";

	WiggleTest wiggleTest;
	int wiggle_test_result = wiggleTest.testAll(wiggleFile1, wiggleFile2, wiggleFile3, wiggleOutFile);
	if (wiggle_test_result != 0)
	{
		std::cerr << "Wiggle test failed!\n";
		if (wiggle_test_result == 1)
		{
			std::cerr << "Invalid repair target - does not conform grammar.\n";
		}
		else if (wiggle_test_result == 2)
		{
			std::cerr << "Object repair attempt failed.\n";
		}
		return wiggle_test_result;
	}
	else
	{
		std::cerr << "Object repair attempt succeeded.\n";
		std::cerr << "Wrote " << wiggleOutFile << "\n";
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

