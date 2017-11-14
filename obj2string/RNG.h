#ifdef _MSC_VER
#pragma once
#endif

#ifndef RNG_H_B4CC193A_3CB4_4B9F_9E9A_8A541BAF045F
#define RNG_H_B4CC193A_3CB4_4B9F_9E9A_8A541BAF045F

#include "Algebra.h"

class KISSRandomNumberGenerator
{
public:
	uint data[4];
	//data[0],
	//data[1],//must be zero
	//data[2],
	//data[3]; //doesn't need to be re-seeded but must be < 698769069

	__host__ __device__ KISSRandomNumberGenerator(
		const uint aX = 123456789u,
		const uint aY = 362436069u,
		const uint aZ = 521288629u,
		const uint aW = 416191069u)
	{
		data[0] = (aX); data[1] = (aY); data[2] = (aZ); data[3] = (aW);
	}

	__host__ __device__ float operator()()
	{
		data[2] = (36969 * (data[2] & 65535) + (data[2] >> 16)) << 16;
		data[3] = 18000 * (data[3] & 65535) + (data[3] >> 16) & 65535;
		data[0] = 69069 * data[0] + 1234567;
		data[1] = (data[1] = (data[1] = data[1] ^ (data[1] << 17)) ^ (data[1] >> 13)) ^ (data[1] << 5);
		return ((data[2] + data[3]) ^ data[0] + data[1]) * 2.328306E-10f;
	}
};

class XorShift32Plus
{
public:
	/* The state must be seeded so that it is not all zero */
	uint s_0;
	uint s_1;

	__host__ __device__ XorShift32Plus(
		const uint aS0,
		const uint aS1)
	{
		s_0 = aS0 == 0u ? 362436069u : aS0;
		s_1 = aS1 == 0u ? 416191069u : aS1;
	}

	__host__ __device__ uint xorshift32plus(void) {
		uint x = s_0;
		uint const y = s_1;
		s_0 = y;
		x ^= x << 13; // a
		s_1 = x ^ y ^ (x >> 17) ^ (y >> 16); // b, c
		return s_1 + y;
	}
	__host__ __device__ float operator()()
	{
		return (float)(xorshift32plus() % 1073741824) * 9.313226E-10f;
	}
};


class HaltonNumberGenerator
{

public:
	const float mPrimesRCP[11] = { 0.5f, 0.333333f, 0.2f, 0.142857f,
		0.09090909f, 0.07692307f, 0.058823529f, 0.0526315789f, 0.04347826f,
		0.034482758f, 0.032258064f };

	__device__ __host__ float operator()(const int aSeed, const int aDimension) const
	{
		if (aDimension < 11)
		{
			float res = 0.f;
			float basisRCP = mPrimesRCP[aDimension];
			const float BASISRCP = mPrimesRCP[aDimension];
			float seed = static_cast<float>(aSeed);

			while (seed)
			{
				float tmp = seed * BASISRCP;
#ifdef __CUDA_ARCH___
				seed = truncf(tmp);
#else
				seed = static_cast<float>(static_cast<int>(tmp));
#endif
				res += basisRCP * (tmp - seed);
				basisRCP *= mPrimesRCP[aDimension];

			}

			return res;
		}
		else
		{
			return 2.f;
		}

	}
};

#endif //RNG_H_B4CC193A_3CB4_4B9F_9E9A_8A541BAF045F
