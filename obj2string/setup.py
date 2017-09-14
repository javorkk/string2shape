from distutils.core import setup, Extension

# the c++ extension module
extension_mod = Extension("obj_tools", ["obj2string.cpp"], extra_compile_args=['/EHsc'],)

setup(name = "obj_tools", ext_modules=[extension_mod])