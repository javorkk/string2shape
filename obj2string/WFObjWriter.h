#ifdef _MSC_VER
#pragma once
#endif

#ifndef OBJWRITER_H_INCLUDED_D7EDAF11_214A_4C74_92D9_74E7A4B0DB6D
#define OBJWRITER_H_INCLUDED_D7EDAF11_214A_4C74_92D9_74E7A4B0DB6D

#include <iostream>
#include <fstream>
#include <string>


class ObjWriter
{

public:
	std::ofstream objFileStream;
	std::ofstream materialFileStream;
	std::ofstream instancesFileStream;

	void init(const char* aFilename);
	void writeVertex(float aX, float aY, float aZ);
	void writeVertexNormal(float aX, float aY, float aZ);
	void writeObjectHeader(int aObjId);
	void writeObjectHeader(size_t aObjId, const char* aMaterialName);
	void writeTriangleIndices(int aA, int aB, int aC);
	void writeTriangleIndices(size_t aA, size_t aB, size_t aC, size_t aNormalA, size_t aNormalB, size_t aNormalC);
	void writePointIndex(int aA);
	void writeLineIndices(int aA, int aB);
	void writeDiffuseMaterial(int aId, float aR, float aG, float aB);
	void writeMaterial(const char* aMaterialName, float aR, float aG, float aB);
	void writeInstance(
		float aT00, float aT10, float aT20, float aT30, /*transformation matrix row 0 (rotation x 3, translation) */
		float aT01, float aT11, float aT21, float aT31, /*transformation matrix row 1 (rotation x 3, translation */
		float aT02, float aT12, float aT22, float aT32, /*transformation matrix row 2 (rotation x 3, translation */
														/*transformation matrix row 3 is 0 0 0 1 */
		int aObjectId,
		float aMinX, float aMinY, float aMinZ, //min bound
		float aMaxX, float aMaxY, float aMaxZ //max bound
	);
	void cleanup();
};


#endif // OBJWRITER_H_INCLUDED_D7EDAF11_214A_4C74_92D9_74E7A4B0DB6D
