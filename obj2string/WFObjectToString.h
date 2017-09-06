#ifdef _MSC_VER
#pragma once
#endif

#ifndef WFOBJECTTOSTRING_H_E4391BD7_542A_47E4_8CC6_351328F4805B
#define WFOBJECTTOSTRING_H_E4391BD7_542A_47E4_8CC6_351328F4805B

#ifdef __cplusplus
extern "C" {
#endif

__declspec(dllexport) extern char * WFObjectToString(const char * aFilename);

__declspec(dllexport) extern char * WFObjectToStrings(const char * aFilename);

__declspec(dllexport) extern char * WFObjectRandomVariations(const char * aFilename1, const char* aFilename2);

__declspec(dllexport) extern int buildGrid(const char * aFilename, int aResX, int aResY, int aResZ);

__declspec(dllexport) extern int testGraphConstruction(int aGraphSize);

__declspec(dllexport) extern int testCollisionGraphConstruction(const char * aFilename);

__declspec(dllexport) extern int testRandomVariations(const char * aFilename1, const char* aFilename2);

__declspec(dllexport) extern int fixVariation(const char * aFileName1, const char* aFileName2, const char* aFileName3, const char* aOutFileName);

__declspec(dllexport) extern int testRandomNumberGenerator(void);


#ifdef __cplusplus
}
#endif

#endif //WFOBJECTTOSTRING_H_E4391BD7_542A_47E4_8CC6_351328F4805B
