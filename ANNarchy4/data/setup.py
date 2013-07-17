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
                                 language="c++")]
    )
else:
    setup(
        cmdclass = { 'build_ext' : build_ext },
        ext_modules = [ Extension("ANNarchyCython", ["pyx/ANNarchy.pyx"], libraries=["annarchy-3.1d"], language="c++")]
    )
