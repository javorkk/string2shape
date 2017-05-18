#ifdef _MSC_VER
#pragma once
#endif

#ifndef TIMER_H_33374EFC_A528_4DBB_B101_7A4B4F52E987
#define TIMER_H_33374EFC_A528_4DBB_B101_7A4B4F52E987

#include <thrust/detail/config.h>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#	include <cuda_runtime_api.h>
#	include <iostream>
#	include <string>
#else
#	include <ctime>
#endif


namespace cudastd
{
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
	struct timer
	{
		cudaEvent_t startT, endT;

		timer()
		{
			myCudaSafeCall(cudaEventCreate(&startT));
			myCudaSafeCall(cudaEventCreate(&endT));
			start();
		}

		void cleanup()
		{
			myCudaSafeCall(cudaEventDestroy(startT));
			myCudaSafeCall(cudaEventDestroy(endT));
		}

		void start()
		{

			myCudaSafeCall(cudaEventRecord(startT, 0));
		}

		float get()
		{
			float elapsedTime;
			myCudaSafeCall(cudaEventRecord(endT, 0));
			myCudaSafeCall(cudaEventSynchronize(endT));
			myCudaSafeCall(cudaEventElapsedTime(&elapsedTime, startT, endT));
			return elapsedTime;
		}

		void myCudaSafeCall(cudaError_t error, const std::string& message = "")
		{
			if (error)
			{
				std::cerr << "Error while timing: " << cudaGetErrorString(error) <<"\n";
			}
		}
	};

#else
	struct timer
	{
		clock_t startT, endT;
		timer()
		{
			start();
		}

		void cleanup() {}

		void start()
		{
			startT = clock();
		}

		float get()
		{
			float elapsedTime;
			endT = clock();
			elapsedTime = 1000.f * static_cast<float>(static_cast<double>(endT - startT) / static_cast<double>(CLOCKS_PER_SEC));
			return elapsedTime;
		}
	};

#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

}//namespace cudastd


#endif // TIMER_H_33374EFC_A528_4DBB_B101_7A4B4F52E987

