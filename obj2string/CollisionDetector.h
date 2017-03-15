#ifdef _MSC_VER
#pragma once
#endif

#ifndef COLLISIONDETECTOR_H_5B5089A7_F465_4E03_BC7E_A719C57C50B8
#define COLLISIONDETECTOR_H_5B5089A7_F465_4E03_BC7E_A719C57C50B8

#include "WFObject.h"
#include "Graph.h"

class CollisionDetector
{
	float totalTime;
	float initTime;
	float trimmTime;
	float adjMatTime;
	float countTime;
	float writeTime;
	float sortTime;
	float uniqueTime;
	float graphTime;

public:
	__host__ Graph computeCollisionGraph(WFObject& aObj, float aRelativeThreshold);
	__host__ void stats();
};


#endif // COLLISIONDETECTOR_H_5B5089A7_F465_4E03_BC7E_A719C57C50B8
