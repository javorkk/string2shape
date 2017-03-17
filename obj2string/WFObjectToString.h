#ifdef _MSC_VER
#pragma once
#endif

#ifndef WFOBJECTTOSTRING_H_E4391BD7_542A_47E4_8CC6_351328F4805B
#define WFOBJECTTOSTRING_H_E4391BD7_542A_47E4_8CC6_351328F4805B

#ifdef __cplusplus
extern "C" {
#endif

__declspec(dllexport) extern const char * WFObjectToString(char * aFilename);

__declspec(dllexport) extern int buildGrid(const char * aFilename, int aResX, int aResY, int aResZ);

__declspec(dllexport) extern int testGraphConstruction(int aGraphSize);

__declspec(dllexport) extern int testCollisionGraphConstruction(const char * aFilename);


#ifdef __cplusplus
}
#endif

#endif //WFOBJECTTOSTRING_H_E4391BD7_542A_47E4_8CC6_351328F4805B
