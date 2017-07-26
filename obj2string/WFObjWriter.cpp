#include "pch.h"
#include "WFObjWriter.h"

//#define DUMP_INSTANCES

void ObjWriter::init(const char * aFilename)
{
	std::string aFileNameStr(aFilename);
	objFileStream.open((aFileNameStr + ".obj").c_str(), std::ios::binary | std::ios::out);

	if (!objFileStream)
	{
		std::cerr << "Could not open file " << aFileNameStr << ".obj for writing\n";
	}

	objFileStream.precision(12);

	materialFileStream.open((aFileNameStr + ".mtl").c_str(), std::ios::binary | std::ios::out);
	if (!materialFileStream)
	{
		std::cerr << "Could not open file " << aFileNameStr << ".mtl for writing\n";
	}

#ifdef DUMP_INSTANCES
	instancesFileStream.open((aFileNameStr + ".obji").c_str(), std::ios::binary | std::ios::out);
	if (!instancesFileStream)
	{
		std::cerr << "Could not open file " << aFileNameStr << ".obji for writing\n";
	}
#endif

	std::string fileNameMtl = aFileNameStr + ".mtl";
	if (fileNameMtl.find_last_of("/\\") == std::string::npos)
		fileNameMtl = fileNameMtl.substr(0, fileNameMtl.size() - 5);
	else
		fileNameMtl = fileNameMtl.substr(fileNameMtl.find_last_of("/\\") + 1, fileNameMtl.size() - fileNameMtl.find_last_of("/\\") - 5);


	objFileStream << "mtllib ";
	objFileStream << fileNameMtl;
	objFileStream << ".mtl\n";
}

void ObjWriter::writeVertex(float aX, float aY, float aZ)
{
	objFileStream << "v " << aX << " " << aY << " " << aZ << "\n";
}

void ObjWriter::writeVertexNormal(float aX, float aY, float aZ)
{
	objFileStream << "vn " << aX << " " << aY << " " << aZ << "\n";
}

void ObjWriter::writeObjectHeader(int aObjId)
{
	objFileStream << "o " << aObjId << "\n";
	objFileStream << "usemtl id_" << aObjId << "\n";
}

void ObjWriter::writeObjectHeader(size_t aObjId, const char* aMaterialName)
{
	objFileStream << "o " << aObjId << "\n";
	objFileStream << "usemtl " << aMaterialName << "\n";
}

void ObjWriter::writeTriangleIndices(int aA, int aB, int aC)
{
	objFileStream << "f ";
	objFileStream << aA + 1 << " ";
	objFileStream << aB + 1 << " ";
	objFileStream << aC + 1 << "\n";
}

void ObjWriter::writeTriangleIndices(size_t aA, size_t aB, size_t aC, size_t aNormalA, size_t aNormalB, size_t aNormalC)
{
	objFileStream << "f ";
	objFileStream << aA + 1 << "//" << aNormalA + 1 << " ";
	objFileStream << aB + 1 << "//" << aNormalB + 1 << " ";
	objFileStream << aC + 1 << "//" << aNormalC + 1 << "\n";
}

void ObjWriter::writePointIndex(int aA)
{
	objFileStream << "p ";
	objFileStream << aA + 1 << "\n";
}

void ObjWriter::writeLineIndices(int aA, int aB)
{
	objFileStream << "l ";
	objFileStream << aA + 1 << " ";
	objFileStream << aB + 1 << "\n";
}

void ObjWriter::writeDiffuseMaterial(int aId, float aR, float aG, float aB)
{
	materialFileStream << "\n";
	materialFileStream << "newmtl id_" << aId << "\n";
	materialFileStream << "illum 2\n";
	materialFileStream << "Ka 0 0 0\n";
	materialFileStream << "Kd " << aR << " " << aG << " " << aB << "\n";
	materialFileStream << "Ks 0 0 0\n";
	materialFileStream << "Ke 0 0 0\n";
	materialFileStream << "Ni 1.0\n";
	materialFileStream << "\n";
}

void ObjWriter::writeMaterial(const char* aMaterialName, float aR, float aG, float aB)
{
	materialFileStream << "\n";
	materialFileStream << "newmtl " << aMaterialName << "\n";
	materialFileStream << "illum 2\n";
	materialFileStream << "Ka 0 0 0\n";//TODO
	materialFileStream << "Kd " << aR << " " << aG << " " << aB << "\n";
	materialFileStream << "Ks 0 0 0\n";//TODO
	materialFileStream << "Ke 0 0 0\n";//TODO
	materialFileStream << "Ni 1.0\n";//TODO
	materialFileStream << "\n";
}

void ObjWriter::writeInstance(
	float aT00, float aT10, float aT20, float aT30, /*transformation matrix row 0 (rotation x 3, translation) */
	float aT01, float aT11, float aT21, float aT31, /*transformation matrix row 1 (rotation x 3, translation */
	float aT02, float aT12, float aT22, float aT32, /*transformation matrix row 2 (rotation x 3, translation */
													/*transformation matrix row 3 is 0 0 0 1 */
	int aObjectId,
	float aMinX, float aMinY, float aMinZ, /*min bound */
	float aMaxX, float aMaxY, float aMaxZ /*max bound */)
{
#ifdef DUMP_INSTANCES
	instancesFileStream << "#new instance#\n";
	instancesFileStream << "obj_id " << aObjectId << "\n";
	instancesFileStream << "m_row_0 " << aT00 << " " << aT10 << " " << aT20 << " " << aT30 << "\n";
	instancesFileStream << "m_row_1 " << aT01 << " " << aT11 << " " << aT21 << " " << aT31 << "\n";
	instancesFileStream << "m_row_2 " << aT02 << " " << aT12 << " " << aT22 << " " << aT32 << "\n";
	instancesFileStream << "AABB_min " << aMinX << " " << aMinY << " " << aMinZ << "\n";
	instancesFileStream << "AABB_max " << aMaxX << " " << aMaxY << " " << aMaxZ << "\n";
#endif
}

void ObjWriter::cleanup()
{
	objFileStream.close();
	materialFileStream.close();

#ifdef DUMP_INSTANCES
	instancesFileStream.close();
#endif
}
