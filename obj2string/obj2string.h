#ifdef _MSC_VER
#pragma once
#endif

#ifndef OBJ2STRING_H_7D6828E8_013E_4C22_AA13_672FDA8BE07B
#define OBJ2STRING_H_7D6828E8_013E_4C22_AA13_672FDA8BE07B

//#ifdef __cplusplus
//extern "C" {
//#endif
//
//	/* Header file for spammodule */
//
//	/* C API functions */
//#define PyOBJ2String_System_NUM 0
//#define PyOBJ2String_System_RETURN int
//#define PyOBJ2String_System_PROTO (const char *command)
//
//	/* Total number of C API pointers */
//#define PyOBJ2String_API_pointers 1
//	
//#ifdef OBJ2STRING_MODULE
//	/* This section is used when compiling obj2string.c */
//
//	static PyOBJ2String_System_RETURN PyOBJ2String_System PyOBJ2String_System_PROTO;
//
//#else
//	/* This section is used in modules that use obj2string's API */
//
//	static void **PySpam_API;
//
//#define PyOBJ2String_System (*(PyOBJ2String_System_RETURN (*)PyOBJ2String_System_PROTO) PyOBJ2String_API[PyOBJ2String_System_NUM])
//
//	/* Return -1 on error, 0 on success.
//	* PyCapsule_Import will set an exception if there's an error.
//	*/
//	static int
//		import_obj2string(void)
//	{
//		PyOBJ2String_API = (void **)PyCapsule_Import("obj2string._C_API", 0);
//		return (PyOBJ2String_API != NULL) ? 0 : -1;
//	}
//
//#endif
//
//#ifdef __cplusplus
//}
//#endif



#endif // OBJWRITER_H_7D6828E8_013E_4C22_AA13_672FDA8BE07B