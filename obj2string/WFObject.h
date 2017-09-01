#ifdef _MSC_VER
#pragma once
#endif

#ifndef FWOBJECT_HPP_INCLUDED_45834E1E_1D79_4F3E_ABD3_77E318EF9223
#define FWOBJECT_HPP_INCLUDED_45834E1E_1D79_4F3E_ABD3_77E318EF9223

#include "Algebra.h"

//A Wavefront3D object. It provides the functionality of 
//	an indexed triangle mesh. The primitive consists
//	of 5 buffers holding vertex positions, vertex normals,
//	vertex texture coordinates, face materials, and the faces
//	them selfs. Each face is a triangle and only contains indices
//	into the position, normal, texture coordinate, and material
//	buffers.
#include <string>
#include <vector>

class WFObject
{
public:
    std::string  name;
    static const float3 C0;
    static const float3 C1;
    static const float3 C099;

    //Represents a material
    struct Material
    {
        //The unique name of the material
		std::string name;
        //diffuse reflectance
        float3 diffuseCoeff;
        //specular reflectance
        float3 specularCoeff;
        //ambient color
        float3 ambientCoeff;
        //emission (light source only)
        float3 emission;

        float specularExp;
        float indexOfRefraction;
        bool isRefractive;


        Material() :
            diffuseCoeff(C0),
            specularCoeff(C0),
            ambientCoeff(C0),
            emission(C0),
            specularExp(1.f),
            indexOfRefraction(1.f),
            isRefractive(false) //non-transparent
        {
			name = "þ no name";
		}
        Material(const std::string& _name) :
            diffuseCoeff(C0),
            specularCoeff(C0),
            ambientCoeff(C0),
            specularExp(1.f),
            indexOfRefraction(1.f),
            isRefractive(false)
        {
			name = _name;
		}

        void setupPhongCoefficients();

    };


    //A face definition (triangle)
    class Face
    {
        WFObject *m_lwObject;
    public:
        size_t material;
        size_t vert1, tex1, norm1;
        size_t vert2, tex2, norm2;
        size_t vert3, tex3, norm3;

        Face() : m_lwObject(NULL)
        {}

        Face(WFObject * _obj) : m_lwObject(_obj) {}
    };

    class Line
    {
        WFObject *m_lwObject;
    public:
        size_t material;
        size_t vert1, tex1, norm1;
        size_t vert2, tex2, norm2;

        Line() : m_lwObject(NULL)
        {}

        Line(WFObject * _obj) : m_lwObject(_obj) {}
    };

    class Point
    {
        WFObject *m_lwObject;
    public:
        size_t material;
        size_t vert1, tex1, norm1;

        Point() : m_lwObject(NULL)
        {}

        Point(WFObject * _obj) : m_lwObject(_obj) {}
    };

    struct Instance
    {
        size_t objectId;
        //Transformation
        float m00, m10, m20, m30;
        float m01, m11, m21, m31;
        float m02, m12, m22, m32;
        //bounding box
        float3 min;
        float3 max;
    };

    typedef std::vector<float3> t_vecVector;
    typedef std::vector<WFObject::Face> t_FaceVector;
    typedef std::vector<WFObject::Line> t_LineVector;
    typedef std::vector<WFObject::Point> t_PointVector;
    typedef std::vector<WFObject::Material> t_materialVector;
    typedef std::vector<float2> t_texCoordVector;
    typedef std::vector<int2> t_objectVector;
    typedef std::vector<WFObject::Instance> t_instancesVector;

    t_vecVector vertices;
    t_vecVector normals;
    t_FaceVector faces;
    t_LineVector lines;
    t_PointVector points;
    t_materialVector materials;
    t_texCoordVector texCoords;
    t_objectVector objects;
    t_instancesVector instances;

    size_t getNumVertices() const
    {
        return vertices.size();
    }
    size_t getNumNormals() const
    {
        return normals.size();
    }
    size_t getNumFaces() const
    {
        return faces.size();
    }
    size_t getNumLines() const
    {
        return lines.size();
    }
    size_t getNumPoints() const
    {
        return points.size();
    }
    size_t getNumMaterials() const
    {
        return materials.size();
    }
    size_t getNumTexCoords() const
    {
        return texCoords.size();
    }

    size_t getNumObjects() const
    {
        return objects.size();
    }
    size_t getNumInstances() const
    {
        return instances.size();
    }

    float3 getVertex(size_t aVtxId) const
    {
        return vertices[aVtxId];
    }

    float3 getNormal(size_t aNormalId) const
    {
        return normals[aNormalId];
    }

    Face getFace(size_t aFaceId) const
    {
        return faces[aFaceId];
    }

    Line getLine(size_t aId) const
    {
        return lines[aId];
    }

    Point getPoint(size_t aId) const
    {
        return points[aId];
    }

    uint getVertexIndex(size_t aId) const;

    uint getNormalIndex(size_t aId) const;

    Material getMaterial(size_t aMatId) const
    {
        return materials[aMatId];
    }

    float2 getTexCoords(size_t aCoordId) const
    {
        return texCoords[aCoordId];
    }

    float3& getVertex(size_t aVtxId)
    {
        return vertices[aVtxId];
    }


    float3& getNormal(size_t aNormalId)
    {
        return normals[aNormalId];
    }

    Face& getFace(size_t aFaceId)
    {
        return faces[aFaceId];
    }

    Material& getMaterial(size_t aMatId)
    {
        return materials[aMatId];
    }

    float2& getTexCoords(size_t aCoordId)
    {
        return texCoords[aCoordId];
    }

    int2 getObjectRange(size_t aId) const
    {
        return objects[aId];
    }

    Instance& getInstance(size_t aInstanceId)
    {
        return instances[aInstanceId];
    }

    const Instance& getInstance(size_t aInstanceId) const
    {
        return instances[aInstanceId];
    }

    size_t insertVertex(const float3& aVertex);

    size_t insertNormal(const float3& aNormal);

    size_t insertFace(const Face& aFace);

    size_t insertLine(const Line& aLine);

    size_t insertPoint(const Point& aPoint);


    size_t insertMaterial(const Material& aMaterial);

    //Reads the WaveFront3D object from a file
    void read(const char* aFileName);
    void loadWFObj(const char* aFileName);
    void loadInstances(const char* aFileName);
	
	void reorderMaterials();

};
#endif // FWOBJECT_HPP_INCLUDED_45834E1E_1D79_4F3E_ABD3_77E318EF9223
