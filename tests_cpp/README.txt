# Run CPP unittests.

For the CPP unittests we rely on Google's testbench framework (see https://google.github.io/googletest/ for details).

Note we do not ship their code with our repository. We use CMake (see https://cmake.org/ for details) to download the
necessary files to a local folder, therefore no extra installation is required (aside from CMake > 1.17). The workflow
is like the following:

mkdir build
cd build
cmake -S .. -B .
make
./ANNarchy_CPP_Templates

Note the *build* folder is optional, but as the whole process generates many files, it's better to place them in a distinct folder (also 
for easier delete of the files), in particular when you also you generate the coverage report described in the next section.

# Code coverage

As for Python, also for CPP unittests, one can generate a coverage report. But this requires an additional tool to generate the report.
We would recommend the *gcovr* Python package (e.g., pip install gcovr). The report is then generate by the following command. We apply
also a filter to ignore uninteresting files (e.g., googles testbench files):

gcovr -r .. --html --html-details -o coverage.html --filter '.*ANNarchy/include/.*'

# A Note on CUDA

Some of the template classes are dedicated for usage on GPU devices using the CUDA framework. However, we will not perform dedicated
tests here, as the CUDA template classes (denoted by [SpMV format]CUDA) are only inherit the corresponding formats (i.e., [SpMV format]),
and add a code for transferring the host representation on the GPU device. The accessor logic etc. is always implemented on the host
side only.