"""
CMakeLists.txt templates.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

# Linux, Seq or OMP
linux_omp_template = """# CMakeLists.txt generated by ANNarchy
cmake_minimum_required(VERSION 3.16)

set(MODULE_NAME "ANNarchyCore%(net_id)s")
set(CMAKE_CXX_COMPILER "%(compiler)s")

project(${MODULE_NAME} LANGUAGES CXX)

# Find Python and add include paths
find_package(Python COMPONENTS Interpreter Development NumPy)
if (Python_FOUND)
include_directories(
    ${Python_INCLUDE_DIRS}
    ${Python_NumPy_INCLUDE_DIRS}
)
endif()

# Detect the installed nanobind package and import it into CMake
# see also: https://nanobind.readthedocs.io/en/latest/building.html
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Additional paths (ANNarchy-related)
include_directories(
    %(annarchy_include)s
    %(thirdparty_include)s
)

# Additional compiler flags (-fPIC will is added already)
add_compile_options(%(cpu_flags)s)
add_link_options(%(cpu_flags)s)

# Compile source files and generate shared library
nanobind_add_module(
    # Target name
    ${MODULE_NAME}
    NB_DOMAIN ${MODULE_NAME}
    # source files
    ${MODULE_NAME}.cpp
    ANNarchy.cpp
)

# Check if OpenMP is available and enable it.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${MODULE_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()

# Set the required C++ standard
target_compile_features(${MODULE_NAME} PUBLIC cxx_std_14)

# Nanobind generates by default ${MODULE_NAME}.cpython[version]-...
# we need to shorten that for easier handling.
set_target_properties(${MODULE_NAME} PROPERTIES SUFFIX ".so")

# After successful compilation move the shared library
add_custom_command(
    TARGET ${MODULE_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${MODULE_NAME}.so ../../
)
"""

# Linux, CUDA
linux_cuda_template = """# CMakeLists.txt generated by ANNarchy
cmake_minimum_required(VERSION 3.16)

set(MODULE_NAME "ANNarchyCore%(net_id)s")
set(CMAKE_CXX_COMPILER "%(compiler)s")

if (CMAKE_VERSION GREATER_EQUAL 3.18)
%(set_cuda_arch)s
endif()

project(${MODULE_NAME} LANGUAGES CXX CUDA)

# Find Python and add include paths
find_package(Python COMPONENTS Interpreter Development NumPy)
if (Python_FOUND)
include_directories(
    ${Python_INCLUDE_DIRS}
    ${Python_NumPy_INCLUDE_DIRS}
)
endif()

# Detect the installed nanobind package and import it into CMake
# see also: https://nanobind.readthedocs.io/en/latest/building.html
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Additional paths (ANNarchy-related)
include_directories(
    %(annarchy_include)s
    %(thirdparty_include)s
)

# For lower CMake versions, setting these flags leads to strange
# behavior (HD: 29th July 2024)
if (CMAKE_VERSION GREATER_EQUAL 3.17)
    # Additional compiler flags (-fPIC will is added already)
    add_compile_options(%(cpu_flags)s)
    add_link_options(%(cpu_flags)s)
endif()

# Compile source files and generate shared library
nanobind_add_module(
    # Target name
    ${MODULE_NAME}
    NB_DOMAIN ${MODULE_NAME}
    # source files (will trigger above command)
    ${MODULE_NAME}.cpp
    ANNarchy.cpp
    ANNarchyKernel.cu
)

# Set the required C++ standard
target_compile_features(${MODULE_NAME} PUBLIC cxx_std_14)

# Nanobind generates by default ${MODULE_NAME}.cpython[version]-...
# we need to shorten that for easier handling.
set_target_properties(${MODULE_NAME} PROPERTIES SUFFIX ".so")

# After successful compilation move the shared library
add_custom_command(
    TARGET ${MODULE_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${MODULE_NAME}.so ../../
)
"""

# OSX, with clang, Seq only
osx_clang_template = """# CMakeLists.txt generated by ANNarchy
cmake_minimum_required(VERSION 3.16)

set(MODULE_NAME "ANNarchyCore%(net_id)s")
set(CMAKE_CXX_COMPILER "%(compiler)s")

project(${MODULE_NAME} LANGUAGES CXX)

# Find Python and add include paths
find_package(Python COMPONENTS Interpreter Development NumPy)
if (Python_FOUND)
include_directories(
    ${Python_INCLUDE_DIRS}
    ${Python_NumPy_INCLUDE_DIRS}
)
endif()

# Detect the installed nanobind package and import it into CMake
# see also: https://nanobind.readthedocs.io/en/latest/building.html
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Additional paths (ANNarchy-related)
include_directories(
    %(annarchy_include)s
    %(thirdparty_include)s
)

# Additional compiler flags (-fPIC will is added already)
add_compile_options(%(cpu_flags)s)
add_link_options(%(cpu_flags)s)

# Compile source files and generate shared library
nanobind_add_module(
    # Target name
    ${MODULE_NAME}
    NB_DOMAIN ${MODULE_NAME}
    # source files
    ${MODULE_NAME}.cpp
    ANNarchy.cpp
)

# Check if OpenMP is available and enable it.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${MODULE_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()

# Set the required C++ standard
target_compile_features(${MODULE_NAME} PUBLIC cxx_std_14)

# Nanobind generates by default ${MODULE_NAME}.cpython[version]-...
# we need to shorten that for easier handling.
set_target_properties(${MODULE_NAME} PROPERTIES SUFFIX ".dylib")

# After successful compilation move the shared library
add_custom_command(
    TARGET ${MODULE_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${MODULE_NAME}.dylib ../../
)
"""

# OSX, with gcc, OpenMP
osx_gcc_template = """# CMakeLists.txt generated by ANNarchy
cmake_minimum_required(VERSION 3.16)

set(MODULE_NAME "ANNarchyCore%(net_id)s")
set(CMAKE_CXX_COMPILER "%(compiler)s")

project(${MODULE_NAME} LANGUAGES CXX)

# Find Python and add include paths
find_package(Python COMPONENTS Interpreter Development NumPy)
if (Python_FOUND)
include_directories(
    ${Python_INCLUDE_DIRS}
    ${Python_NumPy_INCLUDE_DIRS}
)
endif()

# Detect the installed nanobind package and import it into CMake
# see also: https://nanobind.readthedocs.io/en/latest/building.html
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Additional paths (ANNarchy-related)
include_directories(
    %(annarchy_include)s
    %(thirdparty_include)s
)

# Additional compiler flags (-fPIC will is added already)
add_compile_options(%(cpu_flags)s)
add_link_options(%(cpu_flags)s)

# Compile source files and generate shared library
nanobind_add_module(
    # Target name
    ${MODULE_NAME}
    NB_DOMAIN ${MODULE_NAME}
    # source files
    ${MODULE_NAME}.cpp
    ANNarchy.cpp
)

# Check if OpenMP is available and enable it.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${MODULE_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()

# Set the required C++ standard
target_compile_features(${MODULE_NAME} PUBLIC cxx_std_14)

# Nanobind generates by default ${MODULE_NAME}.cpython[version]-...
# we need to shorten that for easier handling.
set_target_properties(${MODULE_NAME} PROPERTIES SUFFIX ".dylib")

# After successful compilation move the shared library
add_custom_command(
    TARGET ${MODULE_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${MODULE_NAME}.dylib ../../
)
"""
