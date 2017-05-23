#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#include "WFObjectToString.h"
#include "UniformGridSortBuilderTest.h"
#include "GraphTest.h"
#include "CollisionTest.h"
#include "ShapeVariationTest.h"
#include "RNGTest.h"

#include <thrust/detail/config.h>

int main()
{

#if 0 //THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
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


	const char* variationFile1 = "../scenes/church/test/variant_c19.obj";
	const char* variationFile2 = "../scenes/church/test/variant_c28.obj";
	//const char* variationFile1 = "../scenes/skyscraper/test/variant_100.obj";
	//const char* variationFile2 = "../scenes/skyscraper/test/variant_101.obj";
	//const char* variationFile1 = "../scenes/playground/variant_000.obj";
	//const char* variationFile2 = "../scenes/playground/variant_001.obj";

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




#if 0// THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
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

