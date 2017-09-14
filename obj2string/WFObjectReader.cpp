#include "pch.h"
#include "WFObject.h"

#ifndef _WIN32
#include <libgen.h>
#define _strnicmp strncasecmp
//patch for gcc-4.3
#include <stdlib.h>
#include <string.h>
#endif

#ifdef _WIN32
#define _PATH_SEPARATOR '\\'
#else
#define _PATH_SEPARATOR '/'
#endif

const float3 WFObject::C0 = rep(0.f);
const float3 WFObject::C1 = rep(1.f);
const float3 WFObject::C099 = rep(0.99f);

void WFObject::Material::setupPhongCoefficients()
{
    specularCoeff = min(specularCoeff, C099 -
        min(C099, diffuseCoeff)) *
        (specularExp + 2) * 0.5f / static_cast<float>(M_PI);

    diffuseCoeff = min(diffuseCoeff, C099)
        / static_cast<float>(M_PI);
}

std::map<std::string, size_t> materialMap;

namespace objLoaderUtil
{
    typedef std::map<std::string, size_t> t_materialMap;

    void skipWS(const char * &aStr)
    {
        while (isspace(*aStr))
            aStr++;
    }

    std::string endSpaceTrimmed(const char* _str)
    {
        size_t len = strlen(_str);
        const char *firstChar = _str;
        const char *lastChar = firstChar + len - 1;
        while (lastChar >= firstChar && isspace(*lastChar))
            lastChar--;

        return std::string(firstChar, lastChar + 1);
    }


//    std::string getDirName(const std::string& _name)
//    {
//        std::string objDir;
//#if _MSC_VER >= 1400
//        char fileDir[4096];
//        _splitpath_s(_name.c_str(), NULL, 0, fileDir, sizeof(fileDir), NULL, 0, NULL, 0);
//        objDir = fileDir;
//#endif
//
//#ifndef _WIN32
//        char *fnCopy = strdup(_name.c_str());
//        const char* dirName = dirname(fnCopy);
//        objDir = dirName;
//        objDir.append("/");
//        free(fnCopy);
//        //std::cerr << "Dirname: " << objDir << "\n";
//
//#endif // _WIN32
//
//        return objDir;
//    }

    std::string skipComments(const std::string& aLine,
        const char* aToken)
    {
        std::string token(aToken);
        const size_t commentStart = aLine.find(token, 0);

        if (commentStart == std::string::npos)
        {
            //line does not contain comment token
            const size_t lineBegin = aLine.find_first_not_of(" \t", 0);
            if (lineBegin < aLine.size())
            {
                return aLine.substr(lineBegin, aLine.size());
            }
            else
            {
                //line contains only whitespace
                return std::string("");
            }
    }
        else
        {
            //line contains comment token
            const size_t lineBegin = aLine.find_first_not_of(" \t", 0);
            return aLine.substr(lineBegin, commentStart - lineBegin);
        }
}



    void readMtlLib(const std::string &aFileName, WFObject::t_materialVector &aMatVector, const t_materialMap &aMatMap)
    {
        std::ifstream matInput(aFileName.c_str(), std::ios::in | std::ios::binary);
        std::string buf;

        if (matInput.fail())
            throw std::runtime_error("Error opening .mtl file");

        size_t curMtl = -1, curLine = 0;

        while (!matInput.eof())
        {
            std::getline(matInput, buf);
            curLine++;

            buf = skipComments(buf, "#");

            if (buf.size() < 1)
            {
                continue;
            }

            const char* cmd = buf.c_str();
            skipWS(cmd);

            if (_strnicmp(cmd, "newmtl", 6) == 0)
            {
                cmd += 6;

                skipWS(cmd);
                std::string name = endSpaceTrimmed(cmd);
                if (aMatMap.find(name) == aMatMap.end())
                    goto parse_err_found;

                curMtl = aMatMap.find(name)->second;
            }
            else if (
                _strnicmp(cmd, "Kd", 2) == 0 || _strnicmp(cmd, "Ks", 2) == 0
                || _strnicmp(cmd, "Ka", 2) == 0 || _strnicmp(cmd, "Ke", 2) == 0)
            {
                char coeffType = *(cmd + 1);

                if (curMtl == -1)
                    goto parse_err_found;

                float3 color = rep(0.f);
                cmd += 2;

                char *newCmdString;
                for (int i = 0; i < 3; i++)
                {
                    skipWS(cmd);
                    ((float*)&color)[i] = (float)strtod(cmd, &newCmdString);
                    if (newCmdString == cmd) goto parse_err_found;
                    cmd = newCmdString;
                }


                switch (coeffType)
                {
                case 'd':
                    aMatVector[curMtl].diffuseCoeff = color; break;
                case 'a':
                    aMatVector[curMtl].ambientCoeff = color; break;
                case 's':
                    aMatVector[curMtl].specularCoeff = color; break;
                case 'e':
                    aMatVector[curMtl].emission = color; break;

                }
            }
            else if (_strnicmp(cmd, "Ns", 2) == 0)
            {
                if (curMtl == -1)
                    goto parse_err_found;

                cmd += 2;

                char *newCmdString;
                skipWS(cmd);
                float coeff = (float)strtod(cmd, &newCmdString);
                if (newCmdString == cmd) goto parse_err_found;
                cmd = newCmdString;
                aMatVector[curMtl].specularExp = coeff;
            }
            else if (_strnicmp(cmd, "Ni", 2) == 0)
            {
                if (curMtl == -1)
                    goto parse_err_found;

                cmd += 2;

                char *newCmdString;
                skipWS(cmd);
                float coeff = (float)strtod(cmd, &newCmdString);
                if (newCmdString == cmd) goto parse_err_found;
                cmd = newCmdString;
                aMatVector[curMtl].indexOfRefraction = coeff;
            }
            else if (_strnicmp(cmd, "illum", 5) == 0)
            {
                if (curMtl == -1)
                    goto parse_err_found;

                cmd += 5;

                char *newCmdString;
                skipWS(cmd);
                long coeff = (long)strtol(cmd, &newCmdString, 10);
                if (newCmdString == cmd) goto parse_err_found;
                cmd = newCmdString;
                if (coeff > 3)
                    aMatVector[curMtl].isRefractive = true;


            }

            continue;
        parse_err_found:
            std::cerr << "Error at line " << curLine << " in " << aFileName << std::endl;
        }
    }
}

using namespace objLoaderUtil;

void WFObject::read(const char* aFileName)
{
    std::string fileName(aFileName);
    std::string fileExtension = fileName.substr(fileName.find_last_of("."), fileName.size());

    vertices.clear();
    normals.clear();
    faces.clear();
    lines.clear();
    points.clear();
    materials.clear();
    texCoords.clear();
    materialMap.clear();
    objects.clear();

    Material defaultMaterial;
    defaultMaterial.name = "Default";
    defaultMaterial.diffuseCoeff = rep(.6774f);
    defaultMaterial.specularCoeff = rep(0.f);
    defaultMaterial.specularExp = 1.f;
    defaultMaterial.ambientCoeff = rep(0.2f);
    defaultMaterial.emission = rep(0.f);
    materials.push_back(defaultMaterial);

    materialMap.insert(std::make_pair(defaultMaterial.name, (size_t)0));

    if (fileExtension == std::string(".obj"))
    {
        loadWFObj(aFileName);
        //normalize reflectance coefficients
        //for (auto it = materials.begin(); it != materials.end(); ++it)
        //{
        //    it->setupPhongCoefficients();
        //}
    }
    else
    {
        std::cerr << "Unknown file extension \"" << fileExtension << "\".\n";
    }

    if (fileName.find_last_of("/\\") == std::string::npos)
    {
        name = fileName.substr(0, fileName.size() - 5);
    }
    else
    {
        name = fileName.substr(fileName.find_last_of("/\\") + 1, fileName.size() - fileName.find_last_of("/\\") - 5);
    }
}

void WFObject::loadWFObj(const char* aFileName)
{

    std::ifstream inputStream(aFileName, std::ios::in);
    std::string buf;

    const size_t _MAX_BUF = 8192;
    const size_t _MAX_IDX = _MAX_BUF / 2;

    float tmpVert[4];
    tmpVert[3] = 0.f;
    size_t tmpIdx[_MAX_IDX * 3];
    int tmpVertPointer, tmpIdxPtr, vertexType;
    size_t curMat = 0, curLine = 0;
    std::vector<std::string> matFiles;

    int currentObjectBegin = (int)faces.size();

    if (inputStream.fail())
    {
        std::cerr << "Error opening .obj file: " << aFileName <<"\n";
        return;
    }

    while (!inputStream.eof())
    {
        std::getline(inputStream, buf);
        buf = skipComments(buf, "#");

        if (buf.size() < 1)
        {
            continue;
        }

        const char *cmdString = buf.c_str();

        curLine++;
        skipWS(cmdString);
        switch (tolower(*cmdString))
        {
        case 0:
            break;
        case 'v':
            cmdString++;
            switch (tolower(*cmdString))
            {
            case 'n': vertexType = 1; cmdString++; break;
            case 't': vertexType = 2; cmdString++; break;
            default:
                if (isspace(*cmdString))
                    vertexType = 0;
                else
                    goto parse_err_found;
            }

            tmpVertPointer = 0;
            for (;;)
            {
                skipWS(cmdString);
                if (*cmdString == 0)
                    break;

                char *newCmdString;
                float flt = (float)strtod(cmdString, &newCmdString);
                if (newCmdString == cmdString)
                    goto parse_err_found;

                cmdString = newCmdString;

                if (tmpVertPointer >= sizeof(tmpVert) / sizeof(float))
                    goto parse_err_found;

                tmpVert[tmpVertPointer++] = flt;
            }

            if (vertexType != 2 && tmpVertPointer != 3 || vertexType == 2 && tmpVertPointer < 2)
                goto parse_err_found;


            if (vertexType == 0)
            {
                vertices.push_back(*(float3*)tmpVert);
            }
            else if (vertexType == 1)
            {
                normals.push_back(*(float3*)tmpVert);
            }
            else
                texCoords.push_back(*(float2*)tmpVert);

            break;

        case 'f':
        case 'l':
            cmdString++;
            if (tolower(*cmdString) == 'o')
                cmdString++;
            skipWS(cmdString);

            tmpIdxPtr = 0;
            for (;;)
            {
                if (tmpIdxPtr + 3 >= sizeof(tmpIdx) / sizeof(int))
                    goto parse_err_found;

                char *newCmdString;
                int idx = strtol(cmdString, &newCmdString, 10);

                if (cmdString == newCmdString)
                    goto parse_err_found;

                cmdString = newCmdString;

                tmpIdx[tmpIdxPtr++] = idx - 1;

                skipWS(cmdString);

                if (*cmdString == '/')
                {
                    cmdString++;

                    skipWS(cmdString);
                    if (*cmdString != '/')
                    {
                        idx = strtol(cmdString, &newCmdString, 10);

                        if (cmdString == newCmdString)
                            goto parse_err_found;

                        cmdString = newCmdString;

                        tmpIdx[tmpIdxPtr++] = idx - 1;
                    }
                    else
                        tmpIdx[tmpIdxPtr++] = -1;


                    skipWS(cmdString);
                    if (*cmdString == '/')
                    {
                        cmdString++;
                        skipWS(cmdString);
                        idx = strtol(cmdString, &newCmdString, 10);

                        //Do ahead lookup of one number
                        skipWS((const char * &)newCmdString);
                        if (isdigit(*newCmdString) || (*newCmdString == 0 || *newCmdString == '#') && cmdString != newCmdString)
                        {
                            if (cmdString == newCmdString)
                                goto parse_err_found;

                            cmdString = newCmdString;

                            tmpIdx[tmpIdxPtr++] = idx - 1;
                        }
                        else
                            tmpIdx[tmpIdxPtr++] = -1;
                    }
                    else
                        tmpIdx[tmpIdxPtr++] = -1;
                }
                else
                {
                    tmpIdx[tmpIdxPtr++] = -1;
                    tmpIdx[tmpIdxPtr++] = -1;
                }

                skipWS(cmdString);
                if (*cmdString == 0)
                    break;
            }

            if (tmpIdxPtr == 3)
            {
                Point p(this);
                p.material = curMat;
                memcpy(&p.vert1, tmpIdx, 3 * sizeof(size_t));

                points.push_back(p);
                break;
            }
            else if (tmpIdxPtr == 6)
            {
                Line l(this);
                l.material = curMat;
                memcpy(&l.vert1, tmpIdx, 6 * sizeof(size_t));

                lines.push_back(l);
                break;
            }
            else if (tmpIdxPtr <= 6)
                goto parse_err_found;

            for (int idx = 3; idx < tmpIdxPtr - 3; idx += 3)
            {
                Face t(this);
                t.material = curMat;
                memcpy(&t.vert1, tmpIdx, 3 * sizeof(size_t));
                memcpy(&t.vert2, tmpIdx + idx, 6 * sizeof(size_t));

                faces.push_back(t);
            }
            break;

        case 'o':
            cmdString++;
            skipWS(cmdString);

            if (currentObjectBegin < faces.size())
            {
                objects.push_back(make_int2(currentObjectBegin, (int)faces.size()));

                char *objNewCmdString;
                if (strtol(cmdString, &objNewCmdString, 10) != objects.size())
                {
                    //std::cerr << "Warning at line " << curLine << ": non-consecutive object index will be ignored!" << std::endl;
                }
            }
            currentObjectBegin = (int)faces.size();

            break;

        case 'g':
        case 's': //?
        case '#':
            //Not supported
            break;

        default:
            if (_strnicmp(cmdString, "usemtl", 6) == 0)
            {
                cmdString += 6;
                skipWS(cmdString);
                std::string name = endSpaceTrimmed(cmdString);
                if (name.empty())
                    goto parse_err_found;

                if (materialMap.find(name) == materialMap.end())
                {
					materialMap[name] = materials.size();
                    materials.push_back(Material(name));
                }

                curMat = materialMap[name];
            }
            else if (_strnicmp(cmdString, "mtllib", 6) == 0)
            {
                cmdString += 6;
                skipWS(cmdString);
                std::string name = endSpaceTrimmed(cmdString);
                if (name.empty())
                    goto parse_err_found;

                matFiles.push_back(name);
            }
            else
            {
                std::cerr << "Unknown entity at line " << curLine << std::endl;
            }
        }

        continue;
    parse_err_found:
        std::cerr << "Error at line " << curLine << std::endl;
    }

    objects.push_back(make_int2(currentObjectBegin, (int)faces.size()));

    std::string objDir = getDirName(aFileName);

    for (std::vector<std::string>::const_iterator it = matFiles.begin(); it != matFiles.end(); it++)
    {
        std::string mtlFileName = objDir + *it;

        readMtlLib(mtlFileName, materials, materialMap);
    }

	if (normals.size() <= 0u)
	{
		for (auto it = faces.begin(); it != faces.end(); it++)
		{
			it->norm1 = it->norm2 = it->norm3 = (size_t)-1;
		}
	}

    for (auto it = faces.begin(); it != faces.end(); it++)
    {
        if (it->norm1 == -1 || it->norm2 == -1 || it->norm3 == -1)
        {
            float3 e1 = vertices[it->vert2] - vertices[it->vert1];
            float3 e2 = vertices[it->vert3] - vertices[it->vert1];
            float3 n = ~(e1 % e2);
            if (it->norm1 == -1) it->norm1 = normals.size();
            if (it->norm2 == -1) it->norm2 = normals.size();
            if (it->norm3 == -1) it->norm3 = normals.size();

            normals.push_back(n);
        }
    }

    for (auto it = lines.begin(); it != lines.end(); it++)
    {
        if (it->norm1 == -1 || it->norm2 == -1)
        {
            float3 n = make_float3(0.f, 0.f, 0.f);
            if (it->norm1 == -1) it->norm1 = normals.size();
            if (it->norm2 == -1) it->norm2 = normals.size();

            normals.push_back(n);
        }
    }

    for (auto it = points.begin(); it != points.end(); it++)
    {
        if (it->norm1 == -1)
        {
            float3 n = make_float3(0.f, 0.f, 0.f);
            if (it->norm1 == -1) it->norm1 = normals.size();

            normals.push_back(n);
        }
    }

	reorderMaterials();
}

uint WFObject::getVertexIndex(size_t aId) const
{
    size_t result = (uint)-1;
    if (faces.size() > 0u)
    {
        switch (aId % 3)
        {
        case 0:
            result = faces[aId / 3].vert1;
            break;
        case 1:
            result = faces[aId / 3].vert2;
            break;
        case 2:
            result = faces[aId / 3].vert3;
            break;
        default:
            break;
        }
    }
    else if (lines.size() > 0u)
    {
        switch (aId % 2)
        {
        case 0:
            result = lines[aId / 2].vert1;
            break;
        case 1:
            result = lines[aId / 2].vert2;
            break;
        default:
            break;
        }
    }
    else if (points.size() > 0u)
    {
        result = points[aId].vert1;
    }

    return (uint)result;
}

uint WFObject::getNormalIndex(size_t aId) const
{
    size_t result = (uint)-1;
    if (faces.size() > 0u)
    {
        switch (aId % 3)
        {
        case 0:
            result = faces[aId / 3].norm1;
            break;
        case 1:
            result = faces[aId / 3].norm2;
            break;
        case 2:
            result = faces[aId / 3].norm3;
            break;
        default:
            break;
        }
    }
    else if (lines.size() > 0u)
    {
        switch (aId % 2)
        {
        case 0:
            result = lines[aId / 2].norm1;
            break;
        case 1:
            result = lines[aId / 2].norm2;
            break;
        default:
            break;
        }
    }
    else if (points.size() > 0u)
    {
        result = points[aId].norm1;
    }

    return (uint)result;
}

size_t WFObject::insertVertex(const float3& aVertex)
{
    vertices.push_back(aVertex);
    return vertices.size() - 1u;
}

size_t WFObject::insertNormal(const float3& aNormal)
{
    normals.push_back(aNormal);
    return normals.size() - 1u;
}

size_t WFObject::insertFace(const Face& aFace)
{
    faces.push_back(aFace);
    return faces.size() - 1u;
}

size_t WFObject::insertLine(const Line & aLine)
{
    lines.push_back(aLine);
    return lines.size() - 1u;
}

size_t WFObject::insertPoint(const Point& aPoint)
{
    points.push_back(aPoint);
    return points.size() - 1u;
}


size_t WFObject::insertMaterial(const Material& aMaterial)
{
    materials.push_back(aMaterial);
    return materials.size() - 1u;
}

void WFObject::loadInstances(const char* aFileName)
{
    std::ifstream input(aFileName, std::ios::in);

    if (input.fail())
        std::cerr << "Could not open file " << aFileName << " for reading.\n"
        << __FILE__ << __LINE__ << std::endl;

    std::string line, buff;
    instances.clear();

    Instance newInstance;
    newInstance.objectId = 0u;
    newInstance.m00 = newInstance.m11 = newInstance.m22 = 1.f;
    newInstance.m01 = newInstance.m02 = newInstance.m12 = 0.f;
    newInstance.m10 = newInstance.m20 = newInstance.m21 = 0.f;
    newInstance.m30 = newInstance.m31 = newInstance.m32 = 0.f;
    newInstance.min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    newInstance.min = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    while (!input.eof())
    {
        std::getline(input, line);

        line = skipComments(line, "#");

        if (line.size() < 1)
        {
            continue;
        }

        std::stringstream ss(line);
        ss >> buff;

        if (buff == "obj_id")
        {
            //start new instance
            instances.push_back(newInstance);
            ss >> instances.back().objectId;
        }
        else if (buff == "m_row_0")
        {
            ss >> instances.back().m00 >> instances.back().m10 >> instances.back().m20 >> instances.back().m30;
        }
        else if (buff == "m_row_1")
        {
            ss >> instances.back().m01 >> instances.back().m11 >> instances.back().m21 >> instances.back().m31;
        }
        else if (buff == "m_row_2")
        {
            ss >> instances.back().m02 >> instances.back().m12 >> instances.back().m22 >> instances.back().m32;
        }
        else if (buff == "AABB_min")
        {
            ss >> instances.back().min.x >> instances.back().min.y >> instances.back().min.z;
        }
        else if (buff == "AABB_max")
        {
            ss >> instances.back().max.x >> instances.back().max.y >> instances.back().max.z;
        }
    }

    input.close();
}

struct MaterialCompare
{
	bool operator()(const WFObject::Material& aM1, const WFObject::Material& aM2) const
	{
		return aM1.name < aM2.name;
	}
};

void WFObject::reorderMaterials()
{
	WFObject::t_materialVector initial_materials(materials.begin(), materials.end());
	std::sort(materials.begin() + 1, materials.end(), MaterialCompare());
	std::vector<size_t> material_reindex(materials.size());
	material_reindex[0] = 0u;

	for (size_t i = 1u; i < initial_materials.size(); ++i)
	{
		WFObject::Material& aM1 = initial_materials[i];
		for (size_t j = 1; j < materials.size(); ++j)
		{
			WFObject::Material& aM2 = materials[j];
			if (aM1.name == aM2.name)
			{
				material_reindex[i] = j;
				break;
			}
		}
	}

	for (auto faceIt = faces.begin(); faceIt != faces.end(); ++faceIt)
	{
		faceIt->material = material_reindex[faceIt->material];
	}
	for (auto lineIt = lines.begin(); lineIt != lines.end(); ++lineIt)
	{
		lineIt->material = material_reindex[lineIt->material];
	}
	for (auto pointIt = points.begin(); pointIt != points.end(); ++pointIt)
	{
		pointIt->material = material_reindex[pointIt->material];
	}
}
