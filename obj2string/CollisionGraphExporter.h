#ifdef _MSC_VER
#pragma once
#endif

#ifndef COLLISIONGRAPHEXPORTER_H_6518E656_6B14_4564_A650_BFFC9BF068D9
#define COLLISIONGRAPHEXPORTER_H_6518E656_6B14_4564_A650_BFFC9BF068D9

#include "WFObject.h"
#include "Graph.h"

class CollisionGraphExporter
{
	float totalTime;
public:
	__host__ void exportCollisionGraph(const char* aFilePath, WFObject& aObj, Graph& aGraph);
	__host__ void exportSubGraph(const char* aFilePath, WFObject& aObj, Graph& aGraph, size_t aId,const thrust::host_vector<unsigned int>& aNodeFlags);
	__host__ void stats();

};


#endif // COLLISIONGRAPHEXPORTER_H_6518E656_6B14_4564_A650_BFFC9BF068D9
