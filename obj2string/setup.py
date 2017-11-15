from distutils.core import setup, Extension
import sys

# the c++ extension module
extension_mod = Extension("obj_tools", [sys.path[0]+"/obj2string.cpp"])

setup(name = "obj_tools", ext_modules=[extension_mod])
