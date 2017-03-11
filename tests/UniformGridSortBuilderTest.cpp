#include "UniformGridSortBuilderTest.h"
#include "WFObjectToString.h"

__host__ int  UniformGridSortBuildTest::testAll(
	const char * aFileName, 
	const int aResX, 
	const int aResY, 
	const int aResZ)
{
	
	return buildGrid(aFileName, aResX, aResY, aResZ);

}
