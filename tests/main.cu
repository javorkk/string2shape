#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#include "WFObjectToString.h"
#include "UniformGridSortBuilderTest.h"
#include "GraphTest.h"
#include "CollisionTest.h"

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
	collTest.testAll("../scenes/castle/castle.obj");

	std::cerr << "Collision detection test passed. \n";

	std::cerr << "---------------------------------------------------------------------\n";
	const char* obj2strTestFile = "../scenes/church/church.obj";
	std::cerr << obj2strTestFile << " converted to \n"
		<< WFObjectToString(obj2strTestFile) << "\n";


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

