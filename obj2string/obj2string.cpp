#include "pch.h"
#include "WFObjectToString.h"

#ifdef __cplusplus
extern "C" {
#endif

const char * obj2string(const char * aFilename)
{	
	const char * retval = WFObjectToString(aFilename);
	return retval;
}

const char * obj2strings(const char * aFilename)
{
	const char * retval = WFObjectToStrings(aFilename);
	return retval;
}

const char* create_variations(const char* aFileName1, const char* aFileName2)
{
	const char * retval = WFObjectRandomVariations(aFileName1, aFileName2);
	return retval;
}

const int fix_variation(const char * aFileName1, const char* aFileName2, const char* aFileName3, const char* aOutFileName)
{
	const int retval = fixVariation(aFileName1, aFileName2, aFileName3, aOutFileName);
	return retval;
}

static PyObject * obj2string_wrapper(PyObject * self, PyObject * args)
{
	const char * result;
	const char * input;
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

static PyObject * obj2strings_wrapper(PyObject * self, PyObject * args)
{
	const char * result;
	const char * input;
	PyObject * ret;

	// parse arguments
	if (!PyArg_ParseTuple(args, "s", &input)) {
		return NULL;
	}

	// run the actual function
	result = obj2strings(input);

	// build the resulting string into a Python object.
	ret = PyUnicode_FromString(result);
	//free(result);

	return ret;
}

static PyObject * create_variations_wrapper(PyObject * self, PyObject * args)
{
	const char * result;
	const char * input1;
	const char * input2;

	PyObject * ret;

	// parse arguments
	if (!PyArg_ParseTuple(args, "s|s", &input1, &input2)) {
		return NULL;
	}

	// run the actual function
	result = create_variations(input1, input2);

	// build the resulting string into a Python object.
	ret = PyUnicode_FromString(result);
	//free(result);

	return ret;
}

static PyObject * fix_variation_wrapper(PyObject * self, PyObject * args)
{
	int result;
	const char * input1;
	const char * input2;
	const char * input3;
	const char * output;


	PyObject * ret;

	// parse arguments
	if (!PyArg_ParseTuple(args, "s|s|s|s", &input1, &input2, &input3, &output)) {
		return NULL;
	}

	// run the actual function
	result = fix_variation(input1, input2, input3, output);

	// build the resulting int into a Python object.
	ret = PyLong_FromLong(result);

	return ret;
}

static PyMethodDef OBJ2StringMethods[] = {
	{ "obj2string", obj2string_wrapper, METH_VARARGS, "Converts a .obj file into a SMILES-type string." },
	{ "obj2strings", obj2strings_wrapper, METH_VARARGS, "Converts a .obj file into multiple SMILES-type strings separated with new lines." },
	{ "create_variations", create_variations_wrapper, METH_VARARGS, "Creates random variations out of a pair of .obj files and writes them in the same folder. Returns the SMILES-type strings representing the variations." },
	{ "fix_variation", fix_variation_wrapper, METH_VARARGS, "Given a pair of example .obj files, attempts to repair a random variations. If successful, the repaired object is written to a file." },
	{ NULL, NULL, 0, NULL }
};

#if PY_MAJOR_VERSION >= 3
/////////////////////////////////////////////////////////
//Python 3.5
/////////////////////////////////////////////////////////
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
#else
/////////////////////////////////////////////////////////
//Python 2.7
/////////////////////////////////////////////////////////
PyMODINIT_FUNC initobj_tools(void)
{
	(void)Py_InitModule("obj_tools", OBJ2StringMethods);
}
#endif

#ifdef __cplusplus
}
#endif