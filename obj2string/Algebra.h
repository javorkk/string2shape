#ifdef _MSC_VER
#pragma once
#endif

#ifndef ALGEBRA_H_6EF7F840_567C_4ED0_9C81_9F3CB81DA890
#define ALGEBRA_H_6EF7F840_567C_4ED0_9C81_9F3CB81DA890

#include <cuda_runtime_api.h>
#include <vector_types.h> //float3
#include <vector_functions.h> //make_float3
#include <math.h> //sqrtf

#ifndef EPS
#define EPS 0.00001f
#endif // !EPS

#ifdef _WIN32
#   define FORCE_INLINE __forceinline__
#else
#   define FORCE_INLINE inline
#endif

#define _DEF_BIN_OP3(_OP)                                                      \
__host__ __device__ FORCE_INLINE float3 operator _OP (const float3& aVal1, const float3& aVal2)                   \
{                                                                              \
    float3 result;                                                             \
    result.x = aVal1.x _OP aVal2.x;                                            \
    result.y = aVal1.y _OP aVal2.y;                                            \
    result.z = aVal1.z _OP aVal2.z;                                            \
    return result;                                                             \
}                                                                              \
__host__ __device__ FORCE_INLINE float3 operator _OP##= (float3& aVal1, const float3& aVal2)          \
{                                                                              \
    aVal1.x = aVal1.x _OP aVal2.x;                                             \
    aVal1.y = aVal1.y _OP aVal2.y;                                             \
    aVal1.z = aVal1.z _OP aVal2.z;                                             \
    return aVal1;                                                              \
}                                                                              \
    /*End Macro */

#define _DEF_UNARY_MINUS3                                                      \
__host__ __device__ FORCE_INLINE float3 operator- (float3 aVal)                                    \
{                                                                              \
    float3 result;                                                             \
    result.x = -aVal.x;                                                        \
    result.y = -aVal.y;                                                        \
    result.z = -aVal.z;                                                        \
    return result;                                                             \
}                                                                              \
    /*End Macro */

#define _DEF_SCALAR_OP3(_OP)                                                   \
__host__ __device__ FORCE_INLINE float3 operator _OP (const float3& aVec, float aVal)                      \
{                                                                              \
    float3 result;                                                             \
    result.x = aVec.x _OP aVal;                                                \
    result.y = aVec.y _OP aVal;                                                \
    result.z = aVec.z _OP aVal;                                                \
    return result;                                                             \
}                                                                              \
__host__ __device__ FORCE_INLINE float3 operator _OP##= (float3& aVec, float aV)                     \
{                                                                              \
    aVec.x _OP##= aV; aVec.y _OP##= aV; aVec.z _OP##= aV;                      \
    return aVec;                                                               \
}

#define _DEF_SCALAR_OP3_SYM(_OP)                                               \
__host__ __device__  FORCE_INLINE float3 operator _OP (float aVal, const float3& aVec)                     \
{                                                                              \
    float3 result;                                                             \
    result.x = aVal _OP aVec.x;                                                \
    result.y = aVal _OP aVec.y;                                                \
    result.z = aVal _OP aVec.z;                                                \
    return result;                                                             \
}                                                                              \
    /*End Macro*/

_DEF_BIN_OP3(+) //Component-wise +
_DEF_BIN_OP3(-) //Component-wise -
_DEF_BIN_OP3(*) //Component-wise *
_DEF_BIN_OP3(/ ) //Component-wise /

_DEF_SCALAR_OP3(*) //Vector * scalar -> Vector and Vector *= scalar
_DEF_SCALAR_OP3_SYM(*) //scalar * Vector -> Vector
_DEF_SCALAR_OP3(/ ) //Vector / scalar and Vector /= scalar

//An unary minus
_DEF_UNARY_MINUS3

//Make a vector with equal components
__host__ __device__ FORCE_INLINE float3 rep(float aVal)
{
	float3 result;
	result.x = result.y = result.z = aVal;
	return result;
}

//Access a vector as array
__host__ __device__ FORCE_INLINE float* toPtr(float3& aVec)
{
	return reinterpret_cast<float*>(&aVec);
}
__host__ __device__ FORCE_INLINE const float* toPtr(const float3& aVec)
{
	return reinterpret_cast<const float*>(&aVec);
}

__host__ __device__ FORCE_INLINE float3 fastDivide(const float3 aVec1, const float3 aVec2)
{
#ifdef __CUDA_ARCH__
	float3 retval;
	retval.x = __fdividef(aVec1.x, aVec2.x);
	retval.y = __fdividef(aVec1.y, aVec2.y);
	retval.z = __fdividef(aVec1.z, aVec2.z);
	return retval;
#else
	return aVec1 / aVec2;
#endif
}

__host__ __device__ FORCE_INLINE float3 fastDivide(const float3& aVec1, const float& aVal)
{
#ifdef __CUDA_ARCH__
	float3 retval;
	retval.x = __fdividef(aVec1.x, aVal);
	retval.y = __fdividef(aVec1.y, aVal);
	retval.z = __fdividef(aVec1.z, aVal);
	return retval;
#else
	return aVec1 / aVal;
#endif
}

//dot product 
__host__ __device__ FORCE_INLINE float dot(const float3& aVec1, const float3& aVec2)
{
	return aVec1.x * aVec2.x + aVec1.y * aVec2.y + aVec1.z * aVec2.z;
}

//dot product with the reciprocal of this
__host__ __device__ FORCE_INLINE float dotRCP(const float3& aVec1, const float3& aVec2)
{
#ifdef __CUDA_ARCH__
	return  __fdividef(aVec1.x, aVec2.x) +
		__fdividef(aVec1.y, aVec2.y) +
		__fdividef(aVec1.z, aVec2.z);
#else
	return aVec1.x / aVec2.x + aVec1.y / aVec2.y + aVec1.z / aVec2.z;
#endif
}

//Cross product on the first three components
__host__ __device__ FORCE_INLINE float3 cross(const float3& aVec1, const float3& aVec2)
{
	float3 retval;
	retval.x = aVec1.y * aVec2.z - aVec1.z * aVec2.y;
	retval.y = aVec1.z * aVec2.x - aVec1.x * aVec2.z;
	retval.z = aVec1.x * aVec2.y - aVec1.y * aVec2.x;
	return retval;
}

//Cross product with the reciprocal of this
__host__ __device__ FORCE_INLINE float3 crossRCP(const float3& aVec1, const float3& aVec2)
{
	float3 retval;
#ifdef __CUDA_ARCH__
	retval.x = __fdividef(aVec2.z, aVec1.y) - __fdividef(aVec2.y, aVec1.z);
	retval.y = __fdividef(aVec2.x, aVec1.z) - __fdividef(aVec2.z, aVec1.x);
	retval.z = __fdividef(aVec2.y, aVec1.x) - __fdividef(aVec2.x, aVec1.y);
#else
	retval.x = aVec2.z / aVec1.y - aVec2.y / aVec1.z;
	retval.y = aVec2.x / aVec1.z - aVec2.z / aVec1.x;
	retval.z = aVec2.y / aVec1.x - aVec2.x / aVec1.y;
#endif
	return retval;
}

//A component-wise maximum between two float3s
__host__ __device__ FORCE_INLINE float3 min(const float3 & aVec1, const float3 & aVec2)
{
#ifdef __CUDA_ARCH__
	float3 retval;
	retval.x = fminf(aVec1.x, aVec2.x);
	retval.y = fminf(aVec1.y, aVec2.y);
	retval.z = fminf(aVec1.z, aVec2.z);
	return retval;
#else
	float3 retval;
	retval.x = (aVec1.x < aVec2.x) ? aVec1.x : aVec2.x;
	retval.y = (aVec1.y < aVec2.y) ? aVec1.y : aVec2.y;
	retval.z = (aVec1.z < aVec2.z) ? aVec1.z : aVec2.z;
	return retval;
#endif
}

__host__ __device__ FORCE_INLINE float3 max(const float3 & aVec1, const float3 & aVec2)
{
#ifdef __CUDA_ARCH__
	float3 retval;
	retval.x = fmaxf(aVec1.x, aVec2.x);
	retval.y = fmaxf(aVec1.y, aVec2.y);
	retval.z = fmaxf(aVec1.z, aVec2.z);
	return retval;
#else
	float3 retval;
	retval.x = (aVec1.x > aVec2.x) ? aVec1.x : aVec2.x;
	retval.y = (aVec1.y > aVec2.y) ? aVec1.y : aVec2.y;
	retval.z = (aVec1.z > aVec2.z) ? aVec1.z : aVec2.z;
	return retval;
#endif
}


//Length of the vector
__host__ __device__ FORCE_INLINE float len(const float3& aVec)
{
	return sqrtf(dot(aVec, aVec));
}

__host__ __device__ FORCE_INLINE float lenRCP(const float3& aVec)
{
#ifdef __CUDA_ARCH__
	return rsqrtf(dot(aVec, aVec));
#else
	return 1.f / sqrtf(dot(aVec, aVec));
#endif
}

//computes orthogonal local coordinate system
__host__ __device__ FORCE_INLINE void getLocalCoordinates(
	const float3& aNormal,
	float3& oTangent,
	float3& oBinormal)
{
	const int cId0 = (fabsf(aNormal.x) > fabsf(aNormal.y)) ? 0 : 1;
	const int cId1 = (fabsf(aNormal.x) > fabsf(aNormal.y)) ? 1 : 0;
	const float sig = (fabsf(aNormal.x) > fabsf(aNormal.y)) ? -1.f : 1.f;

	const float invLen = 1.f / (toPtr(aNormal)[cId0] * toPtr(aNormal)[cId0] +
		aNormal.z * aNormal.z);

	toPtr(oTangent)[cId0] = aNormal.z * sig * invLen;
	toPtr(oTangent)[cId1] = 0.f;
	oTangent.z = toPtr(aNormal)[cId0] * -1.f * sig * invLen;

	oBinormal = cross(aNormal, oTangent);
}

//Cross product
__host__ __device__ FORCE_INLINE float3 operator %(const float3& aVec1, const float3& aVec2)
{
	return cross(aVec1, aVec2);
}

//A normalized vector: V / |V|
__host__ __device__ FORCE_INLINE float3 operator ~(const float3& aVec)
{
	return aVec * lenRCP(aVec);
}

#undef _DEF_BIN_OP3
#undef _DEF_UNARY_MINUS3
#undef _DEF_SCALAR_OP3
#undef _DEF_SCALAR_OP3_SYM

__host__ __device__ FORCE_INLINE float determinant(const float3& col0, const float3& col1, const float3& col2)
{
	return col0.x * col1.y * col2.z + col1.x * col2.y * col0.z + col2.x * col0.y * col1.z -
		col2.x * col1.y * col0.z - col1.x * col0.y * col2.z - col0.x * col2.y * col1.z;
}

__host__ __device__ FORCE_INLINE float determinant(
	float m00, float m10, float m20,
	float m01, float m11, float m21,
	float m02, float m12, float m22
)
{
	return m00 * m11 * m22 + m10 * m21 * m02 + m20 * m01 * m12 -
		m20 * m11 * m02 - m10 * m01 * m22 - m00 * m21 * m12;
}

__host__ __device__ FORCE_INLINE double determinantd(
	double m00, double m10, double m20,
	double m01, double m11, double m21,
	double m02, double m12, double m22
)
{
	return m00 * m11 * m22 + m10 * m21 * m02 + m20 * m01 * m12 -
		m20 * m11 * m02 - m10 * m01 * m22 - m00 * m21 * m12;
}

struct quaternion4f
{
	float x, y, z, w;

	__host__ __device__ FORCE_INLINE quaternion4f() {}

	__host__ __device__ FORCE_INLINE quaternion4f(float aX, float aY, float aZ, float aW) :x(aX), y(aY), z(aZ), w(aW) {}

	__host__ __device__ FORCE_INLINE quaternion4f(
		float m00, float m10, float m20,
		float m01, float m11, float m21,
		float m02, float m12, float m22
	)
	{

		float tr = m00 + m11 + m22;

		if (tr > -EPS)
		{
			const float r = sqrtf(tr + 1.f);
			const float s = 0.5f / r;

			w = 0.5f * r;
			x = (m12 - m21) * s;
			y = (m20 - m02) * s;
			z = (m01 - m10) * s;
		}
		else if ((m00 > m11) && (m00 > m22))
		{
			const float r = sqrtf(1.f + m00 - m11 - m22);
			const float s = 0.5f / r;

			w = (m12 - m21) * s;
			x = 0.5f * r;
			y = (m10 + m01) * s;
			z = (m20 + m02) * s;
		}
		else if (m11 > m22)
		{
			const float r = sqrtf(1.f + m11 - m00 - m22);
			const float s = 0.5f / r;

			w = (m20 - m02) * s;
			x = (m10 + m01) * s;
			y = 0.5f * r;
			z = (m21 + m12) * s;
		}
		else
		{
			const float r = sqrtf(1.f + m22 - m00 - m11);
			const float s = 0.5f / r;
			w = (m01 - m10) * s;
			x = (m20 + m02) * s;
			y = (m21 + m12) * s;
			z = 0.5f * r;
		}
	}

	__host__ __device__ FORCE_INLINE float3 operator ()(const float3& aVec) const
	{
		//const float ww = w * w;
		const float wx = w * x;
		const float wy = w * y;
		const float wz = w * z;

		const float xx = x * x;
		const float xy = x * y;
		const float xz = x * z;


		const float yy = y * y;
		const float yz = y * z;


		const float zz = z * z;


		return make_float3(
			aVec.x - 2.f * (yy + xx) * aVec.x + 2.f * (xy - wz) * aVec.y + 2.f * (xz + wy) * aVec.z,
			aVec.y + 2.f * (xy + wz) * aVec.x - 2.f * (xx + zz) * aVec.y + 2.f * (yz - wx) * aVec.z,
			aVec.z + 2.f * (xz - wy) * aVec.x + 2.f * (yz + wx) * aVec.y - 2.f * (xx + yy) * aVec.z
		);

	}

	__host__ __device__ FORCE_INLINE void toMatrix3f(
		float& oMxx, float& oMyx, float& oMzx,
		float& oMxy, float& oMyy, float& oMzy,
		float& oMxz, float& oMyz, float& oMzz) const
	{
		//const float ww = w * w;
		const float wx = w * x;
		const float wy = w * y;
		const float wz = w * z;

		const float xx = x * x;
		const float xy = x * y;
		const float xz = x * z;


		const float yy = y * y;
		const float yz = y * z;


		const float zz = z * z;

		oMxx = 2.f * (yy + zz); oMyx = 2.f * (xy - wz); oMzx = 2.f * (xz + wy);
		oMxy = 2.f * (xy + wz); oMyy = 2.f * (xx + zz); oMzy = 2.f * (yz - wx);
		oMxz = 2.f * (xz - wy); oMyz = 2.f * (yz + wx); oMzz = 2.f * (xx + yy);
		oMxx = 1.f - oMxx;
		oMyy = 1.f - oMyy;
		oMzz = 1.f - oMzz;

		oMxx = fabsf(oMxx) < EPS ? 0.f : oMxx;
		oMxy = fabsf(oMxy) < EPS ? 0.f : oMxy;
		oMxz = fabsf(oMxz) < EPS ? 0.f : oMxz;
		oMyx = fabsf(oMyx) < EPS ? 0.f : oMyx;
		oMyy = fabsf(oMyy) < EPS ? 0.f : oMyy;
		oMyz = fabsf(oMyz) < EPS ? 0.f : oMyz;
		oMzx = fabsf(oMzx) < EPS ? 0.f : oMzx;
		oMzy = fabsf(oMzy) < EPS ? 0.f : oMzy;
		oMzz = fabsf(oMzz) < EPS ? 0.f : oMzz;

		oMxx = fabsf(1.f - oMxx) < EPS ? 1.f : oMxx;
		oMxy = fabsf(1.f - oMxy) < EPS ? 1.f : oMxy;
		oMxz = fabsf(1.f - oMxz) < EPS ? 1.f : oMxz;
		oMyx = fabsf(1.f - oMyx) < EPS ? 1.f : oMyx;
		oMyy = fabsf(1.f - oMyy) < EPS ? 1.f : oMyy;
		oMyz = fabsf(1.f - oMyz) < EPS ? 1.f : oMyz;
		oMzx = fabsf(1.f - oMzx) < EPS ? 1.f : oMzx;
		oMzy = fabsf(1.f - oMzy) < EPS ? 1.f : oMzy;
		oMzz = fabsf(1.f - oMzz) < EPS ? 1.f : oMzz;

		oMxx = fabsf(1.f + oMxx) < EPS ? -1.f : oMxx;
		oMxy = fabsf(1.f + oMxy) < EPS ? -1.f : oMxy;
		oMxz = fabsf(1.f + oMxz) < EPS ? -1.f : oMxz;
		oMyx = fabsf(1.f + oMyx) < EPS ? -1.f : oMyx;
		oMyy = fabsf(1.f + oMyy) < EPS ? -1.f : oMyy;
		oMyz = fabsf(1.f + oMyz) < EPS ? -1.f : oMyz;
		oMzx = fabsf(1.f + oMzx) < EPS ? -1.f : oMzx;
		oMzy = fabsf(1.f + oMzy) < EPS ? -1.f : oMzy;
		oMzz = fabsf(1.f + oMzz) < EPS ? -1.f : oMzz;

	}

	__host__ __device__ FORCE_INLINE quaternion4f conjugate() const
	{
		return quaternion4f(-x, -y, -z, w);
	}
};

__host__ __device__ FORCE_INLINE quaternion4f make_quaternion4f(float aX, float aY, float aZ, float aW) { return quaternion4f(aX, aY, aZ, aW); }

__host__ __device__ FORCE_INLINE float magnitudeSQR(const quaternion4f& aQ) { return aQ.x * aQ.x + aQ.y * aQ.y + aQ.z * aQ.z + aQ.w * aQ.w; }

__host__ __device__ FORCE_INLINE float magnitude(const quaternion4f& aQ) { return sqrtf(magnitudeSQR(aQ)); }

__host__ __device__ FORCE_INLINE quaternion4f operator /(const quaternion4f& aQ, float aS) { return quaternion4f(aQ.x / aS, aQ.y / aS, aQ.z / aS, aQ.w / aS); }
__host__ __device__ FORCE_INLINE quaternion4f operator *(const quaternion4f& aQ, float aS) { return quaternion4f(aQ.x * aS, aQ.y * aS, aQ.z * aS, aQ.w * aS); }

__host__ __device__ FORCE_INLINE quaternion4f operator ~(const quaternion4f& aQ)
{
	const float magSQR = magnitudeSQR(aQ);
	if (fabsf(magSQR) > EPS && fabsf(magSQR - 1.0f) > EPS) {
		float mag = sqrtf(magSQR);
		return aQ / mag;
	}
	return aQ;
}

__host__ __device__ FORCE_INLINE quaternion4f operator *(const quaternion4f& aQ1, const quaternion4f& aQ2)
{
	return quaternion4f(
		aQ1.w * aQ2.x + aQ1.x * aQ2.w + aQ1.y * aQ2.z - aQ1.z * aQ2.y,
		aQ1.w * aQ2.y + aQ1.y * aQ2.w + aQ1.z * aQ2.x - aQ1.x * aQ2.z,
		aQ1.w * aQ2.z + aQ1.z * aQ2.w + aQ1.x * aQ2.y - aQ1.y * aQ2.x,
		aQ1.w * aQ2.w - aQ1.x * aQ2.x - aQ1.y * aQ2.y - aQ1.z * aQ2.z
	);
}

__host__ __device__ FORCE_INLINE float3 transformVecRCP(const quaternion4f& aQ, const float3& aV_RCP)
{
	quaternion4f result = quaternion4f(
		aQ.w / aV_RCP.x + /*aQ.x / aV_RCP.w +*/ aQ.y / aV_RCP.z - aQ.z / aV_RCP.y,
		aQ.w / aV_RCP.y + /*aQ.y / aV_RCP.w +*/ aQ.z / aV_RCP.x - aQ.x / aV_RCP.z,
		aQ.w / aV_RCP.z + /*aQ.z / aV_RCP.w +*/ aQ.x / aV_RCP.y - aQ.y / aV_RCP.x,
		/*aQ.w * aV_RCP.w*/ -aQ.x / aV_RCP.x - aQ.y / aV_RCP.y - aQ.z / aV_RCP.z
	) * aQ.conjugate();
	return make_float3(result.x, result.y, result.z);
}

__host__ __device__ FORCE_INLINE float3 transformVec(const quaternion4f& aQ, const float3& aV)
{
	quaternion4f result = aQ * quaternion4f(aV.x, aV.y, aV.z, 0.f) * aQ.conjugate();
	return make_float3(result.x, result.y, result.z);
}

__host__ __device__ FORCE_INLINE bool isIdentity(const quaternion4f& aQ, float aEPS)
{
	return fabsf(aQ.x) + fabsf(aQ.y) + fabsf(aQ.z) < aEPS && fabsf(aQ.w - 1.f) < aEPS;
}

__host__ __device__ FORCE_INLINE bool isZero(const quaternion4f& aQ, float aEPS)
{
	return fabsf(aQ.x) + fabsf(aQ.y) + fabsf(aQ.z) + fabsf(aQ.w) < aEPS;
}

struct quaternion3f
{
	float x, y, z;

	__host__ __device__ FORCE_INLINE quaternion3f() {}

	__host__ __device__ FORCE_INLINE quaternion3f(float aX, float aY, float aZ) : x(aX), y(aY), z(aZ)
	{}

	__host__ __device__ FORCE_INLINE quaternion3f(float aX, float aY, float aZ, float aW)
	{
		quaternion4f result = ~make_quaternion4f(aX, aY, aZ, aW);
		x = result.x;
		y = result.y;
		z = result.z;
	}

	__host__ __device__ FORCE_INLINE quaternion3f(const quaternion4f& aQ)
	{
		quaternion4f result = ~aQ;
		x = result.x;
		y = result.y;
		z = result.z;
	}

	__host__ __device__ FORCE_INLINE quaternion3f(
		float m00, float m10, float m20,
		float m01, float m11, float m21,
		float m02, float m12, float m22
	)
	{
		quaternion4f result = ~quaternion4f(
			m00, m10, m20,
			m01, m11, m21,
			m02, m12, m22);

		x = result.x;
		y = result.y;
		z = result.z;
	}

	__host__ __device__ FORCE_INLINE float3 operator ()(const float3& aVec) const
	{
		const float w = sqrtf(1.f - x * x - y * y - z * z);
		quaternion4f expandedQ = make_quaternion4f(x, y, z, w);
		return expandedQ(aVec);
	}

	__host__ __device__ FORCE_INLINE void toMatrix3f(
		float& m00, float& m10, float& m20,
		float& m01, float& m11, float& m21,
		float& m02, float& m12, float& m22) const
	{
		const float w = sqrtf(1.f - x * x - y * y - z * z);

		quaternion4f expandedQ = make_quaternion4f(x, y, z, w);
		expandedQ.toMatrix3f(
			m00, m10, m20,
			m01, m11, m21,
			m02, m12, m22);
	}

	__host__ __device__ FORCE_INLINE quaternion3f conjugate() const
	{
		return quaternion3f(-x, -y, -z);
	}
};

__host__ __device__ FORCE_INLINE quaternion3f operator *(const quaternion3f& aQ1, const quaternion3f& aQ2)
{
	const float aQ1w = sqrtf(1.f - aQ1.x * aQ1.x - aQ1.y * aQ1.y - aQ1.z * aQ1.z);
	const float aQ2w = sqrtf(1.f - aQ2.x * aQ2.x - aQ2.y * aQ2.y - aQ2.z * aQ2.z);
	return quaternion3f(
		aQ1w * aQ2.x + aQ1.x * aQ2w + aQ1.y * aQ2.z - aQ1.z * aQ2.y,
		aQ1w * aQ2.y + aQ1.y * aQ2w + aQ1.z * aQ2.x - aQ1.x * aQ2.z,
		aQ1w * aQ2.z + aQ1.z * aQ2w + aQ1.x * aQ2.y - aQ1.y * aQ2.x//,
																   //aQ1w * aQ2w - aQ1.x * aQ2.x - aQ1.y * aQ2.y - aQ1.z * aQ2.z
	);
}

__host__ __device__ FORCE_INLINE float3 transformVecRCP(const quaternion3f& aQ, const float3& aV_RCP)
{
	const float aQw = sqrtf(1.f - aQ.x * aQ.x - aQ.y * aQ.y - aQ.z * aQ.z);
	quaternion4f result = quaternion4f(
		aQw / aV_RCP.x + /*aQ.x / aV_RCP.w +*/ aQ.y / aV_RCP.z - aQ.z / aV_RCP.y,
		aQw / aV_RCP.y + /*aQ.y / aV_RCP.w +*/ aQ.z / aV_RCP.x - aQ.x / aV_RCP.z,
		aQw / aV_RCP.z + /*aQ.z / aV_RCP.w +*/ aQ.x / aV_RCP.y - aQ.y / aV_RCP.x,
		/*aQw * aV_RCP.w*/ -aQ.x / aV_RCP.x - aQ.y / aV_RCP.y - aQ.z / aV_RCP.z
	) * quaternion4f(-aQ.x, -aQ.y, -aQ.z, aQw);
	return make_float3(result.x, result.y, result.z);
}

__host__ __device__ FORCE_INLINE float3 transformVec(const quaternion3f& aQ, const float3& aV)
{
	const float aQw = sqrtf(1.f - aQ.x * aQ.x - aQ.y * aQ.y - aQ.z * aQ.z);
	quaternion4f result = quaternion4f(aQ.x, aQ.y, aQ.z, aQw) * quaternion4f(aV.x, aV.y, aV.z, 0.f) * quaternion4f(-aQ.x, -aQ.y, -aQ.z, aQw);
	return make_float3(result.x, result.y, result.z);
}

__host__ __device__ FORCE_INLINE bool isIdentity(const quaternion3f& aQ, float aEPS)
{
	return fabsf(aQ.x) + fabsf(aQ.y) + fabsf(aQ.z) < aEPS;
}

__host__ __device__ FORCE_INLINE bool isZero(const quaternion3f& aQ, float aEPS)
{
	return fabsf(aQ.x) + fabsf(aQ.y) + fabsf(aQ.z) < aEPS;
}


struct binary_float3_min
{
	__host__ __device__ float3 operator()(const float3& x, const float3& y) const
	{
		return min(x, y);
	}
};

struct binary_float3_max
{
	__host__ __device__ float3 operator()(const float3& x, const float3& y) const
	{
		return max(x, y);
	}
};

#endif // ALGEBRA_H_6EF7F840_567C_4ED0_9C81_9F3CB81DA890

