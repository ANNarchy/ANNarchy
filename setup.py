#!/usr/bin/env python

################################################
# Check if the required packages are installed
################################################
from __future__ import print_function
import sys, os, os.path, json, shutil
import subprocess
from pkg_resources import parse_version

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


################################################
# Configuration
################################################

def create_config():
    """ Creates config file and check for typical configurations. """
    def cuda_config():
        cuda_path = "/usr/local/cuda"
        if os.path.exists('/usr/local/cuda-7.0'):
            cuda_path = "/usr/local/cuda-7.0"
        if os.path.exists('/usr/local/cuda-8.0'):
            cuda_path = "/usr/local/cuda-8.0"
        if os.path.exists('/usr/local/cuda-9.0'):
            cuda_path = "/usr/local/cuda-9.0"
        if os.path.exists('/usr/local/cuda-10.0'):
            cuda_path = "/usr/local/cuda-10.0"
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
            'flags': "-O3 -march=native",
        }
    elif sys.platform == "darwin":   # mac os
        settings['openmp'] = {
            'compiler': "clang++",
            'flags': "-O3",
        }

    # CUDA settings (optional)
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

# Create the configuration file
settings = create_config()

def python_environment():
    """
    Python environment configuration. Detects among others the python version, library path and cython version.
    """
    # Python version
    py_version = "%(major)s.%(minor)s" % {'major': sys.version_info[0],
                                          'minor': sys.version_info[1]}
    py_major = str(sys.version_info[0])

    if py_major == '2':
        print("WARNING: Python 2 is not supported anymore, things might break.")

    # Python includes and libs
    py_prefix = sys.prefix

    # Search for pythonx.y-config
    cmd = "%(py_prefix)s/bin/python%(py_version)s-config --includes > /dev/null 2> /dev/null"
    with subprocess.Popen(cmd % {'py_version': py_version, 'py_prefix': py_prefix}, shell=True) as test:
        if test.wait() != 0:
            print("Can not find python-config in the same directory as python, trying with the default path...")
            python_config_path = "python%(py_version)s-config"% {'py_version': py_version}
        else:
            python_config_path = "%(py_prefix)s/bin/python%(py_version)s-config" % {'py_version': py_version, 'py_prefix': py_prefix}

    python_include = "`%(pythonconfigpath)s --includes`" % {'pythonconfigpath': python_config_path}
    python_libpath = "-L%(py_prefix)s/lib" % {'py_prefix': py_prefix}

    # Check cython version
    with subprocess.Popen(py_prefix + "/bin/cython%(major)s -V > /dev/null 2> /dev/null" % {'major': py_major}, shell=True) as test:
        if test.wait() != 0:
            cython = py_prefix + "/bin/cython"
        else:
            cython = py_prefix + "/bin/cython" + py_major

    # If not in the same folder as python, use the default
    with subprocess.Popen("%(cython)s -V > /dev/null 2> /dev/null" % {'cython': cython}, shell=True) as test:
        if test.wait() != 0:
            cython = shutil.which("cython"+py_major)
            if cython is None:
                cython = shutil.which("cython")
                if cython is None:
                    print("ERROR: Unable to find the 'cython' executable, fix your $PATH if already installed." )
                    sys.exit(1)
                

    return py_version, py_major, python_include, python_libpath, cython

# Extra compile args
extra_compile_args = ["-O3"]
extra_link_args = []
if sys.platform.startswith('darwin'):
    extra_compile_args.append("-stdlib=libc++")
    extra_link_args = ["-stdlib=libc++"]

################################################
# Perform the installation
################################################
package_data = [
                'core/cython_ext/*.pxd',
                'core/cython_ext/*.pyx',
                'core/cython_ext/CSRMatrix.hpp',
                'include/*.hpp',
                ]

extensions = [
    Extension("ANNarchy.core.cython_ext.Connector",
            ["ANNarchy/core/cython_ext/Connector.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            language="c++"),
    Extension("ANNarchy.core.cython_ext.Coordinates",
            ["ANNarchy/core/cython_ext/Coordinates.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            language="c++"),
    Extension("ANNarchy.core.cython_ext.Transformations",
            ["ANNarchy/core/cython_ext/Transformations.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            language="c++"),
]

dependencies = [
    'numpy',
    'scipy',
    'matplotlib',
    'cython',
    'sympy'
]

release = '4.7.1.6'
print("Installing ANNarchy", release)
py_version, py_major, python_include, python_libpath, cython_major = python_environment()
print("\tPython", py_version, "(", sys.executable, ')')
print("\tIncludes:", python_include)
print("\tLibraries:", python_libpath)
print("\tCython:", cython_major)
print("\tNumpy:", np.get_include())

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
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence'
        ],
        keywords='neural simulator',
        packages=find_packages(),
        package_data={'ANNarchy': package_data},
        install_requires=dependencies,
        ext_modules = cythonize(extensions, language_level=int(sys.version_info[0])),
        include_dirs = [np.get_include()],
        zip_safe = False
)
