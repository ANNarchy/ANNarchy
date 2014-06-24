# Template for creation of the Makefile for OpenMP paradigm
#
# * src_type: set to %.cpp
#
# * obj_type: set to %.o
#
# * flags: additional compiler flags, need to be set (e.g. optimization level)
# 
# The obj_type, src_type flags look artificial and useless, there are two reasons for this:
#
#    * first, every occurance of % is matched as a replacement field and lead to "TypeError: an integer is required"
#    * secondly, through this, we can use one makefile for OpenCL and CUDA, which have in fact different source types
#
omp_makefile = """
# Makefile
SRC = $(wildcard build/*.cpp)
PYX = $(wildcard pyx/*.pyx)
OBJ = $(patsubst build/%(src_type)s, build/%(obj_type)s, $(SRC))
     
all:
\t@echo "Please provide a target, either 'ANNarchyCython_2.6', 'ANNarchyCython_2.7' or 'ANNarchyCython_3.x for python versions."

ANNarchyCython_%(py_version)s: $(OBJ) pyx/ANNarchyCython_%(py_version)s.o
\t@echo "Build ANNarchyCython library for python %(py_version)s"
\tg++ -shared -Wl,-z,relro -fpermissive -std=c++0x -fopenmp build/*.o pyx/ANNarchyCython_%(py_version)s.o -L. -L/usr/lib64 -Wl,-R./annarchy -lpython%(py_version)s %(extra_libs)s -o ANNarchyCython.so  

pyx/ANNarchyCython_%(py_version)s.o : pyx/ANNarchyCython.pyx
\tcython pyx/ANNarchyCython.pyx --cplus  
\tg++ %(flag)s -pipe -Wp,-D_FORTIFY_SOURCE=2 -fexceptions -fstack-protector --param=ssp-buffer-size=4 -D_GNU_SOURCE -fwrapv -fPIC -I/usr/include/python%(py_version)s -I %(numpy_include)s -c pyx/ANNarchyCython.cpp -o pyx/ANNarchyCython_%(py_version)s.o -L. -I. -Ibuild -fopenmp -std=c++0x -fpermissive 

build/%(obj_type)s : build/%(src_type)s
\tg++ %(flag)s -fPIC -pipe -fpermissive -std=c++0x -fopenmp -I. -c $< -o $@

ANNarchyCPP : $(OBJ)
\tg++ %(flag)s -fPIC -shared -fpermissive -std=c++0x -fopenmp -I. build/*.o -o libANNarchyCPP.so

clean:
\trm -rf build/*.o
\trm -rf pyx/*.o
\trm -rf ANNarchyCython.so
"""

# Template for creation of the Makefile for CUDA paradigm
#
# * src_type: set to %.cpp
#
# * src_gpu: set to %.cu (cuda) or %.cl (opencl, but mostly they will be online compiled)
#
# * obj_type: set to %.o
#
# * flags: additional compiler flags, need to be set (e.g. optimization level)
# 
# The obj_type, src_type flags look artificial and useless, there are two reasons for this:
#
#    * first, every occurance of % is matched as a replacement field and lead to "TypeError: an integer is required"
#    * secondly, through this, we can use one makefile for OpenCL and CUDA, which have in fact different source types
#
cuda_makefile = """
# Makefile
SRC = $(wildcard build/*.cpp)
GPU = $(wildcard build/*.cu)
PYX = $(wildcard pyx/*.pyx)
OBJ = $(patsubst build/%(src_type)s, build/%(obj_type)s, $(SRC))
GPU_OBJ = $(patsubst build/%(src_gpu)s, build/%(obj_type)s, $(GPU))
     
all:
\t@echo "Please provide a target, either 'ANNarchyCython_2.6', 'ANNarchyCython_2.7' or 'ANNarchyCython_3.x for python versions or ANNarchyCPP for cpp stand alone version."

#
# python wrapped library
ANNarchyCython_%(py_version)s: $(OBJ) $(GPU_OBJ) pyx/ANNarchyCython_%(py_version)s.o
\t@echo "Build ANNarchyCython library for python %(py_version)s"
\tg++ -shared -Wl,-z,relro -fpermissive -std=c++0x -fopenmp build/*.o pyx/ANNarchyCython_%(py_version)s.o -L. -L/usr/lib64 -Wl,-R./annarchy -lpython%(py_version)s -lcudart -o ANNarchyCython.so  

pyx/ANNarchyCython_%(py_version)s.o : pyx/ANNarchyCython.pyx
\tcython pyx/ANNarchyCython.pyx --cplus  
\tg++ %(flag)s -pipe -Wp,-D_FORTIFY_SOURCE=2 -fexceptions -fstack-protector --param=ssp-buffer-size=4 -D_GNU_SOURCE -fwrapv -fPIC -I/usr/include/python%(py_version)s -c pyx/ANNarchyCython.cpp -o pyx/ANNarchyCython_%(py_version)s.o -L. -I. -Ibuild -fopenmp -std=c++0x -fpermissive 

#
# CPP stand alone library
ANNarchyCPP : $(OBJ) $(GPU_OBJ)
\tg++ %(flag)s -fPIC -shared -fpermissive -std=c++0x -fopenmp -I. build/*.o -lcudart -o libANNarchyCPP.so

#
# compile the source targets
build/%(obj_type)s : build/%(src_type)s
\tg++ %(flag)s -fPIC -pipe -fpermissive -std=c++0x -fopenmp -I. -c $< -o $@

build/%(obj_type)s : build/%(src_gpu)s
\tnvcc -arch=compute_30 -code=sm_30 -Xcompiler -fPIC -I. -c $< -o $@

#
# remove previous builds
clean:
\trm -rf build/*.o
\trm -rf pyx/*.o
\trm -rf ANNarchyCython.so
"""
