#!/usr/bin/env python

################################################
# Check if the required packages are installed
################################################
from __future__ import print_function
import sys, os, os.path, json
from pkg_resources import parse_version
from ANNarchy.generator.Compiler import python_environment
from ANNarchy.generator.Template.MakefileTemplate import cuda_check

# check python version
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


# Check for cuda
has_cuda = False
if os.system("nvcc --version 2> /dev/null") == 0:
    has_cuda = True
    print('Checking for CUDA... OK')
else:
    print('Checking for CUDA... NO')
    print("Warning: CUDA is not available on your system. Only OpenMP can be used to perform the simulations.")


################################################
# Configuration
################################################

def create_config(has_cuda):
    """ Creates config file if not existing already"""
    def cuda_config():
        cuda_path = "/usr/local/cuda"
        if os.path.exists('/usr/local/cuda-7.0'):
            cuda_path = "/usr/local/cuda-7.0"
        if os.path.exists('/usr/local/cuda-8.0'):
            cuda_path = "/usr/local/cuda-8.0"
        if os.path.exists('/usr/local/cuda-9.0'):
            cuda_path = "/usr/local/cuda-9.0"
        return {
            'compiler': "nvcc",
            'flags': "",
            'device': 0,
            'path': cuda_path}

    # Generate a default setting files
    settings = {}

    # OpenMP settings
    if sys.platform.startswith('linux'): # Linux systems
        settings['openmp'] = {
            'compiler': "g++",
            'flags': "-march=native -O2",
        }
    elif sys.platform == "darwin":   # mac os
        settings['openmp'] = {
            'compiler': "clang++",
            'flags': "-march=native -O2",
        }

    # CUDA settings (optional)
    if has_cuda:
        settings['cuda'] = cuda_config()
        
    # If the config file does not exist, create it
    if not os.path.exists(os.path.expanduser('~/.config/ANNarchy/annarchy.json')):
        print('Creating the configuration file in ~/.config/ANNarchy/annarchy.json')
        if not os.path.exists(os.path.expanduser('~/.config/ANNarchy')):
            os.makedirs(os.path.expanduser('~/.config/ANNarchy'))

        # Write the settings
        with open(os.path.expanduser("~/.config/ANNarchy/annarchy.json"), 'w') as f:
            json.dump(settings, f, indent=4)

        return settings

    else: # The config file exists, make sure it has all the required fields
        update_required = False
        with open(os.path.expanduser("~/.config/ANNarchy/annarchy.json"), 'r') as f:
            # Load the existing settings
            local_settings = json.load(f)
            # Check the first level headers exist
            for paradigm in settings.keys():
                if not paradigm in local_settings.keys():
                    local_settings[paradigm] = settings[paradigm]
                    update_required = True
                    break
                # Iterate over all obligatory fields and update then if required
                for key in settings[paradigm].keys():
                    if not key in local_settings[paradigm].keys():
                        local_settings[paradigm][key] = settings[paradigm][key]
                        update_required = True

        if update_required:
            print('Updating the configuration file in ~/.config/ANNarchy/annarchy.json')
            with open(os.path.expanduser("~/.config/ANNarchy/annarchy.json"), 'w') as f:
                json.dump(local_settings, f, indent=4)

        return local_settings

def install_cuda(settings):
    print('Configuring CUDA...')
    # Build the CudaCheck library
    cwd = os.getcwd()
    os.chdir(cwd+"/ANNarchy/generator/CudaCheck")

    # Path to cuda
    cuda_compiler = settings['cuda']['compiler']
    cuda_path = settings['cuda']['path']
    gpu_ldpath = '-L' + cuda_path + '/lib64' +  ' -L' + cuda_path + '/lib'

    # Get the environment
    py_version, py_major, python_include, python_lib, python_libpath, cython_major = python_environment()

    # Write the Makefile to the disk
    with open('Makefile', 'w') as wfile:
        wfile.write(cuda_check % {
            'py_include': python_include,
            'py_libpath': python_libpath,
            'cy_major': cython_major,
            'gpu_compiler': cuda_compiler,
            'gpu_ldpath': gpu_ldpath
            }
        )
    try:
        os.system("make clean && make")
    except:
        print('Something wrong happened when building the CUDA configuration.')
        print('Try setting the correct path to the CUDA installation in ~/.config/ANNarchy/annarchy.json and reinstall.')
    os.chdir(cwd)

# Create the configuration file
settings = create_config(has_cuda)


# Compile the CUDA check module
if has_cuda:
    install_cuda(settings)

################################################
# Perform the installation
################################################
package_data = [
                'core/cython_ext/*.pxd',
                'core/cython_ext/CSRMatrix.hpp',
                'generator/CudaCheck/cuda_check.so',
                'generator/CudaCheck/cuda_check.h',
                'generator/CudaCheck/cuda_check.cu',
                'generator/CudaCheck/cuda_check.pyx',
                'generator/CudaCheck/Makefile'
                ]

extensions = [
    Extension("ANNarchy.core.cython_ext.Connector", 
            ["ANNarchy/core/cython_ext/Connector.pyx"], 
            include_dirs=[np.get_include()], 
            extra_compile_args=["-O2","-std=c++11"]),
    Extension("ANNarchy.core.cython_ext.Coordinates", 
            ["ANNarchy/core/cython_ext/Coordinates.pyx"], 
            include_dirs=[np.get_include()], 
            extra_compile_args=["-O2","-std=c++11"]),
    Extension("ANNarchy.core.cython_ext.Transformations", 
            ["ANNarchy/core/cython_ext/Transformations.pyx"], 
            include_dirs=[np.get_include()], 
            extra_compile_args=["-O2","-std=c++11"]),
]

dependencies = [
    'numpy',
    'scipy',
    'matplotlib',
    'cython',
    'sympy'
]

release = '4.6.6'

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
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
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
