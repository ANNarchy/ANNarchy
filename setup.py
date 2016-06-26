#!/usr/bin/env python

################################################
# Check if the required packages are installed
################################################
from pkg_resources import parse_version

# check python version
import sys, os
if not sys.version_info[:2] >= (2, 7):
    print('Error : ANNarchy requires at least Python 2.7.')
    exit(0)

# setuptools
try:
    from setuptools import setup, find_packages, Extension
    print('Checking for setuptools... OK')
except:
    print('Checking for setuptools... NO')
    print('Error : Python package "setuptools" is required.')
    print('You can install it from pip or: http://pypi.python.org/pypi/setuptools')
    exit(0)

# numpy
try:
    import numpy as np
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
if os.system("nvcc --version 2> /dev/null") == 0:
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
package_data = [
                'core/cython_ext/*.pxd',
                'generator/CudaCheck/cuda_check.so',
                'generator/CudaCheck/cuda_check.h',
                'generator/CudaCheck/cuda_check.cu',
                'generator/CudaCheck/cuda_check.pyx',
                'generator/CudaCheck/Makefile'
                ]

extensions = [
    Extension("ANNarchy.core.cython_ext.Connector", ["ANNarchy/core/cython_ext/Connector.pyx"], include_dirs=[np.get_include()]),
    Extension("ANNarchy.core.cython_ext.Coordinates", ["ANNarchy/core/cython_ext/Coordinates.pyx"], include_dirs=[np.get_include()]),
    Extension("ANNarchy.core.cython_ext.Transformations", ["ANNarchy/core/cython_ext/Transformations.pyx"], include_dirs=[np.get_include()]),
]

dependencies = [
    'numpy',
    'scipy',
    'matplotlib',
    'cython',
    'sympy'
]

release = '4.5.6'

setup(  name='ANNarchy',
        version=release,
        download_url = 'https://bitbucket.org/annarchy/annarchy',
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
        ext_modules = cythonize(extensions),
        include_dirs = [np.get_include()],
        zip_safe = False
)
