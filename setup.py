#!/usr/bin/env python
import sys
import os, os.path
import shutil
import json
import subprocess
import numpy

from setuptools import setup, Extension
from setuptools.command.build import build
from Cython.Build import cythonize

#########################################################
# ANNarchy global configurations (during installation)
#########################################################
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
        print_hint_msg = True
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
                    else:
                        # Check if the set values diverge
                        if local_settings[paradigm][key] != settings[paradigm][key]:
                            if print_hint_msg:
                                print("HINT - we detected diverging settings:")
                                print_hint_msg = False
                            print("   * field ["+paradigm+","+key+"]:")
                            print("       - your annarchy.json: ", local_settings[paradigm][key] )
                            print("       - ANNarchy default: ", settings[paradigm][key] )

        if update_required:
            print('Updating the configuration file in ~/.config/ANNarchy/annarchy.json')
            with open(os.path.expanduser("~/.config/ANNarchy/annarchy.json"), 'w') as f:
                json.dump(local_settings, f, indent=4)

        return local_settings

def python_environment():
    """
    Python environment configuration. Detects among others the python version, library path and cython version.
    """
    # Python version
    py_version = "%(major)s.%(minor)s" % {'major': sys.version_info[0],
                                          'minor': sys.version_info[1]}
    py_major = str(sys.version_info[0])

    # Python includes and libs
    py_prefix = sys.base_prefix

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
                'cython_ext/*.pxd',
                'cython_ext/*.pyx',
                'cython_ext/CSRMatrix.hpp',
                'include/*.hpp',
                'thirdparty/*.hpp'
                ]

extensions = [
    Extension("ANNarchy.cython_ext.Connector",
            ["ANNarchy/cython_ext/Connector.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            language="c++"),
    Extension("ANNarchy.cython_ext.Coordinates",
            ["ANNarchy/cython_ext/Coordinates.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            language="c++"),
    Extension("ANNarchy.cython_ext.Transformations",
            ["ANNarchy/cython_ext/Transformations.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            language="c++"),
]

class CustomizedBuild(build):
    """
    Customization of the build process.
    """

    def run(self):
        """
        Extend the default build step
        """
        # Own stuff
        print("Installing ANNarchy ...")
        py_version, _, python_include, python_libpath, cython_major = python_environment()
        print("\tPython", py_version, "(", sys.executable, ')')
        print("\tIncludes:", python_include)
        print("\tLibraries:", python_libpath)
        print("\tCython:", cython_major)
        print("\tNumpy:", numpy.get_include())

        print("Check annarchy.json")
        create_config()

        print("Building Cython Extensions ...")
        # perform default behavior
        build.run(self)     # NEVER call super, it breaks everything!

# PyExtension stuff remains here while the rest of metadata is contained in
# pyproject.toml ...
setup(
    ext_modules = cythonize(extensions, language_level=int(sys.version_info[0])),
    package_data={'ANNarchy': package_data},
    cmdclass={"build": CustomizedBuild}
)
