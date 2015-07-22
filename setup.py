#!/usr/bin/env python

################################################
# Check if the required packages are installed
################################################
from pkg_resources import parse_version

# check python version
import sys
if not sys.version_info[:2] >= (2, 7):
    print('Error : ANNarchy requires at least Python 2.7.')
    exit(0) 

# setuptools
try:
    from setuptools import setup, find_packages
    print('Checking for setuptools... OK')
except:
    print('Checking for setuptools... NO')
    print('Error : Python package "setuptools" is required.')
    print('You can install it from pip or: http://pypi.python.org/pypi/setuptools')
    exit(0)

# sympy
try:
    import sympy
    if parse_version(sympy.__version__) > parse_version('0.7.4'):
        print('Checking for sympy... OK')
    else:
        print('Sympy ' + str(sympy.__version__) + ' is not sufficient, expected >= 0.7.4' )
        exit(0)
except:
    print('Checking for sympy... NO')
    print('Error : Python package "sympy" >= 0.7.4 is required.')
    print('You can install it from pip or: http://www.sympy.org')
    exit(0)
    
# numpy
try:
    import numpy
    print('Checking for numpy... OK')
except:
    print('Checking for numpy... NO')
    print('Error : Python package "numpy" is required.')
    print('You can install it from pip or: http://www.numpy.org')
    exit(0)

# cython
try:
    import cython
    from Cython.Build import cythonize
    print('Checking for cython... OK')

except:
    print('Checking for cython... NO')
    print('Error : Python package "cython" is required.')
    print('You can install it from pip or: http://www.cython.org')
    exit(0)

# check for cuda
import os
if os.system("nvcc --version")==0:
    # if nvcc available build the CudaCheck library
    cwd = os.getcwd()
    print('Checking for CUDA... OK')
    os.chdir(cwd+"/ANNarchy/generator/CudaCheck")
    os.system("make clean && make")
    os.chdir(cwd)
else:
    print('Checking for CUDA... NO')
    print("Warning: CUDA is not available on your system. Only OpenMP can be used to perform the simulations.")


################################################
# Perform the installation
################################################
import ANNarchy

package_data = ['core/cython_ext/*.pxd','generator/CudaCheck/cuda_check.so']

if sys.platform.startswith("darwin"):
    #   11.02.2015 (hd)
    #   On darwin-based platforms, the distutils package on python 2.x does not work properly. Building of the ext_modules does not chain to clang++, instead the C compiler (clang) is used. As we include e. g. vector classes from cython this lead to compilation errors.
    #   As a solution we cythonize *.pyx and compile the resulting *.cpp to the corresponding shared libraries from hand. Afterwords the created libraries are copied.
    cwd = os.getcwd()
    os.chdir(cwd+"/ANNarchy/core/cython_ext")
    os.system("make clean && make")
    os.chdir(cwd)
    package_data.append('core/cython_ext/*.so')
    ext_modules = None
    dependencies = []
    print('Warning: on  MacOSX, dependencies are not automatically installed when using pip.')
else: # linux
    ext_modules = cythonize(
            [   "ANNarchy/core/cython_ext/Connector.pyx", 
                "ANNarchy/core/cython_ext/Coordinates.pyx",
                "ANNarchy/core/cython_ext/Transformations.pyx"]
        )
    dependencies = [
          'numpy',
          'scipy',
          'matplotlib',
          'cython',
          'sympy'
        ]


setup(  name='ANNarchy',
        version=ANNarchy.__release__,
        download_url = 'https://bitbucket.org/annarchy/annarchy/get/'+ANNarchy.__release__+'.zip',
        license='GPLv2+',
        platforms='GNU/Linux; MacOSX',
        description='Artificial Neural Networks architect',
        long_description="""ANNarchy (Artificial Neural Networks architect) is a parallel simulator for distributed rate-coded or spiking neural networks. The core of the library is generated in C++ and distributed using openMP or CUDA. It provides an interface in Python for the definition of the networks.""",
        author='Julien Vitay, Helge Uelo Dinkelbach and Fred Hamker',
        author_email='julien.vitay@informatik.tu-chemnitz.de',
        url='http://www.tu-chemnitz.de/informatik/KI/projects/ANNarchy/index.php',
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Environment :: Console',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)',
            'Operating System :: POSIX :: Linux',
            'Operating System :: MacOS :: MacOS X',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.3',
            'Programming Language :: Python :: 3.4',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence'
        ],
        keywords='neural simulator',
        packages=find_packages(),
        package_data={'ANNarchy': package_data},
        install_requires=dependencies,
        ext_modules = ext_modules,
        include_dirs = [numpy.get_include()],
        zip_safe = False
)
