from distutils.core import setup, Extension
import sys

# the c++ extension module
obj_tools_mod = Extension(
	"obj_tools",
	define_macros = [("MAJOR_VERSION", "1"), ("MINOR_VERSION", "0")],
	libraries = ["obj2string"],
	sources = [sys.path[0]+"/obj2string.cpp"])

setup(
	name = "obj_tools", 
	version = "1.0",
	long_description = "Geometry manipulation tools for .obj files.",
	ext_modules = [obj_tools_mod]
	)
