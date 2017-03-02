
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "UniformGridSortBuilderTest.h"


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
	uniformGridTest.testAll("../scenes/castle.obj", 32, 32, 32);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

