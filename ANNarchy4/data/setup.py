from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import sys, os

if sys.platform.startswith('linux'):
    setup(
        cmdclass = { 'build_ext' : build_ext },
        ext_modules = [ Extension("ANNarchyCython", 
                                 sources=["pyx/ANNarchyCython.pyx"], 
                                 libraries=["ANNarchyCore"], 
                                 runtime_library_dirs=["./annarchy"],
                                 extra_compile_args=["-fopenmp"],
                                 language="c++")]
    )
else:
    setup(
        cmdclass = { 'build_ext' : build_ext },
        ext_modules = [ Extension("ANNarchyCython", 
                                 sources=["pyx/ANNarchyCython.pyx"], 
                                 include_dirs=[numpy.get_include()],
                                 libraries=["ANNarchyCore"], 
                                 extra_compile_args=["/EHsc", "/I."],
                                 extra_link_args=["/LIBPATH:annarchy"],
                                 language="c++")]
    )
