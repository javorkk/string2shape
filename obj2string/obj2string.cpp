#include "pch.h"
//#define OBJ2STRING_MODULE
#include "WFObjectToString.h"

#ifdef __cplusplus
extern "C" {
#endif

char * obj2string(char * aFilename)
{	
	char * retval = WFObjectToString(aFilename);
	return retval;
}



static PyObject * obj2string_wrapper(PyObject * self, PyObject * args)
{
	char * input;
	char * result;
	PyObject * ret;

	// parse arguments
	if (!PyArg_ParseTuple(args, "s", &input)) {
		return NULL;
	}

	// run the actual function
	result = obj2string(input);

	// build the resulting string into a Python object.
	ret = PyUnicode_FromString(result);
	//free(result);

	return ret;
}


static PyMethodDef OBJ2StringMethods[] = {
	{ "obj2string", obj2string_wrapper, METH_VARARGS, "Say hello" },
	{ NULL, NULL, 0, NULL }
};

static struct PyModuleDef obj2string_module = {
	PyModuleDef_HEAD_INIT,
	"obj_tools",   /* name of module */
	NULL, /* module documentation, may be NULL */
	-1,       /* size of per-interpreter state of the module,
			  or -1 if the module keeps state in global variables. */
	OBJ2StringMethods
};

PyMODINIT_FUNC PyInit_obj_tools(void)
{
	return PyModule_Create(&obj2string_module);
}

#ifdef __cplusplus
}
#endif