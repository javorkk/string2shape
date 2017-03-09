#ifdef _MSC_VER
#pragma once
#endif

#ifndef COLLISIONGRAPHEXPORTER_H_6518E656_6B14_4564_A650_BFFC9BF068D9
#define COLLISIONGRAPHEXPORTER_H_6518E656_6B14_4564_A650_BFFC9BF068D9

#include "WFObject.h"
#include "Graph.h"

class CollisionGraphExporter
{
public:
	void exportCollisionGraph(const char* aFileName, WFObject& aObj, Graph& aGraph) const;
};


#endif // COLLISIONGRAPHEXPORTER_H_6518E656_6B14_4564_A650_BFFC9BF068D9
