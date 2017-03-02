#include "pch.h"
#include "WFObjectToString.h"

#include "WFObject.h"
#include "UniformGrid.h"
#include "UniformGridSortBuilder.h"

#ifdef __cplusplus
extern "C" {
#endif

	char * WFObjectToString(char * aFilename)
	{
		char* testFileName = "scenes/castle.obj";
		WFObject testObj;

		testObj.loadWFObj(testFileName);
	
		UniformGridSortBuilder builder;
		UniformGrid grid = builder.build(testObj, 24, 24, 24);
		
		return testFileName;
	}

	int buildGrid(const char * aFilename, int aResX, int aResY, int aResZ)
	{
		WFObject testObj;
		testObj.loadWFObj(aFilename);

		UniformGridSortBuilder builder;
		UniformGrid grid = 	builder.build(testObj, aResX, aResY, aResZ);

		return 0;
	}


#ifdef __cplusplus
}
#endif