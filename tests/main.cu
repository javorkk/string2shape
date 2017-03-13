
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "UniformGridSortBuilderTest.h"
#include "GraphTest.h"
#include "CollisionTest.h"


int main()
{
	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 1;
	}

	UniformGridSortBuildTest uniformGridTest;
	int ugrid_test_result = uniformGridTest.testAll("../scenes/castle/castle.obj", 32, 16, 24);
	if (ugrid_test_result != 0)
	{
		fprintf(stderr, "Uniform grid construction test failed!\n");
		return ugrid_test_result;
	}
	else
	{
		fprintf(stderr, "Uniform grid construction test passed.\n");
	}

	GraphTest graphTest;
	int graph_test_result = graphTest.testAll(100);
	//graphTest.testAll(1000);

	if (graph_test_result != 0)
	{
		fprintf(stderr, "Graph construction test failed!\n");
		return graph_test_result;
	}
	else
	{
		fprintf(stderr, "Graph construction test passed.\n");
	}

	CollisionTest collTest;
	collTest.testAll("../scenes/castle/castle.obj");
	fprintf(stderr, "Collision detection test passed. (check collision graph)\n");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

