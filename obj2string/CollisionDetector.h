#ifdef _MSC_VER
#pragma once
#endif

#ifndef COLLISIONDETECTOR_H_5B5089A7_F465_4E03_BC7E_A719C57C50B8
#define COLLISIONDETECTOR_H_5B5089A7_F465_4E03_BC7E_A719C57C50B8

#include "WFObject.h"
#include "Graph.h"

class CollisionDetector
{
public:
	Graph computeCollisionGraph(WFObject& aObj, float aRelativeThreshold) const;
};


#endif // COLLISIONDETECTOR_H_5B5089A7_F465_4E03_BC7E_A719C57C50B8
