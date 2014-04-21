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

ANNarchyCython_2.6: $(OBJ) pyx/ANNarchyCython_2.6.o
\t@echo "Build ANNarchyCython library for python 2.6"
\tg++ -shared -Wl,-z,relro -fpermissive -std=c++0x -fopenmp build/*.o pyx/ANNarchyCython_2.6.o -L. -L/usr/lib64 -Wl,-R./annarchy -lpython2.6 -o ANNarchyCython.so  

pyx/ANNarchyCython_2.6.o : pyx/ANNarchyCython.pyx
\tcython pyx/ANNarchyCython.pyx --cplus  
\tg++ %(flag)s -pipe -Wp,-D_FORTIFY_SOURCE=2 -fexceptions -fstack-protector --param=ssp-buffer-size=4 -D_GNU_SOURCE -fwrapv -fPIC -I/usr/include/python2.6 -c pyx/ANNarchyCython.cpp -o pyx/ANNarchyCython_2.6.o -L. -I. -Ibuild -fopenmp -std=c++0x -fpermissive 

ANNarchyCython_2.7: $(OBJ) pyx/ANNarchyCython_2.7.o
\t@echo "Build ANNarchyCython library for python 2.7"
\tg++ -shared -Wl,-z,relro -fpermissive -std=c++0x -fopenmp build/*.o pyx/ANNarchyCython_2.7.o -L. -L/usr/lib64 -Wl,-R./annarchy -lpython2.7 -o ANNarchyCython.so  

pyx/ANNarchyCython_2.7.o : pyx/ANNarchyCython.pyx
\tcython pyx/ANNarchyCython.pyx --cplus  
\tg++ %(flag)s -pipe -Wp,-D_FORTIFY_SOURCE=2 -fexceptions -fstack-protector --param=ssp-buffer-size=4 -D_GNU_SOURCE -fwrapv -fPIC -I/usr/include/python2.7 -c pyx/ANNarchyCython.cpp -o pyx/ANNarchyCython_2.7.o -L. -I. -Ibuild -fopenmp -std=c++0x -fpermissive 

ANNarchyCython_3.x: $(OBJ) pyx/ANNarchyCython_3.x.o
\t@echo "Build ANNarchyCython library for python 3.x"
\tg++ -shared -Wl,-z,relro -fpermissive -std=c++0x -fopenmp build/*.o pyx/ANNarchyCython_3.x.o -L. -L/usr/lib64 -Wl,-R./annarchy -lpython3.2mu -o ANNarchyCython.so  

pyx/ANNarchyCython_3.x.o : pyx/ANNarchyCython.pyx
\tcython pyx/ANNarchyCython.pyx --cplus  
\tg++ %(flag)s -pipe -Wp,-D_FORTIFY_SOURCE=2 -fexceptions -fstack-protector --param=ssp-buffer-size=4 -D_GNU_SOURCE -fwrapv -fPIC -I/usr/include/python3.2 -c pyx/ANNarchyCython.cpp -o pyx/ANNarchyCython_3.x.o -L. -I. -Ibuild -fopenmp -std=c++0x -fpermissive 

build/%(obj_type)s : build/%(src_type)s
\tg++ %(flag)s -fPIC -pipe -fpermissive -std=c++0x -fopenmp -I. -c $< -o $@

ANNarchyCPP : $(OBJ)
\tg++ %(flag)s -fPIC -shared -fpermissive -std=c++0x -fopenmp -I. build/*.o -o libANNarchyCPP.so

clean:
\trm -rf build/*.o
\trm -rf pyx/*.o
\trm -rf ANNarchyCython.so
"""

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
cuda_makefile = """
# Makefile
SRC = $(wildcard build/*.cpp)
PYX = $(wildcard pyx/*.pyx)
OBJ = $(patsubst build/%(src_type)s, build/%(obj_type)s, $(SRC))
     
all:
\t@echo "Please provide a target, either 'ANNarchyCython_2.6', 'ANNarchyCython_2.7' or 'ANNarchyCython_3.x for python versions."

ANNarchyCython_2.6: $(OBJ) pyx/ANNarchyCython_2.6.o
\t@echo "Build ANNarchyCython library for python 2.6"
\tg++ -shared -Wl,-z,relro -fpermissive -std=c++0x -fopenmp build/*.o pyx/ANNarchyCython_2.6.o -L. -L/usr/lib64 -Wl,-R./annarchy -lpython2.6 -o ANNarchyCython.so  

pyx/ANNarchyCython_2.6.o : pyx/ANNarchyCython.pyx
\tcython pyx/ANNarchyCython.pyx --cplus  
\tg++ %(flag)s -pipe -Wp,-D_FORTIFY_SOURCE=2 -fexceptions -fstack-protector --param=ssp-buffer-size=4 -D_GNU_SOURCE -fwrapv -fPIC -I/usr/include/python2.6 -c pyx/ANNarchyCython.cpp -o pyx/ANNarchyCython_2.6.o -L. -I. -Ibuild -fopenmp -std=c++0x -fpermissive 

ANNarchyCython_2.7: $(OBJ) pyx/ANNarchyCython_2.7.o
\t@echo "Build ANNarchyCython library for python 2.7"
\tg++ -shared -Wl,-z,relro -fpermissive -std=c++0x -fopenmp build/*.o pyx/ANNarchyCython_2.7.o -L. -L/usr/lib64 -Wl,-R./annarchy -lpython2.7 -o ANNarchyCython.so  

pyx/ANNarchyCython_2.7.o : pyx/ANNarchyCython.pyx
\tcython pyx/ANNarchyCython.pyx --cplus  
\tg++ %(flag)s -pipe -Wp,-D_FORTIFY_SOURCE=2 -fexceptions -fstack-protector --param=ssp-buffer-size=4 -D_GNU_SOURCE -fwrapv -fPIC -I/usr/include/python2.7 -c pyx/ANNarchyCython.cpp -o pyx/ANNarchyCython_2.7.o -L. -I. -Ibuild -fopenmp -std=c++0x -fpermissive 

ANNarchyCython_3.x: $(OBJ) pyx/ANNarchyCython_3.x.o
\t@echo "Build ANNarchyCython library for python 3.x"
\tg++ -shared -Wl,-z,relro -fpermissive -std=c++0x -fopenmp build/*.o pyx/ANNarchyCython_3.x.o -L. -L/usr/lib64 -Wl,-R./annarchy -lpython3.2mu -o ANNarchyCython.so  

pyx/ANNarchyCython_3.x.o : pyx/ANNarchyCython.pyx
\tcython pyx/ANNarchyCython.pyx --cplus  
\tg++ %(flag)s -pipe -Wp,-D_FORTIFY_SOURCE=2 -fexceptions -fstack-protector --param=ssp-buffer-size=4 -D_GNU_SOURCE -fwrapv -fPIC -I/usr/include/python3.2 -c pyx/ANNarchyCython.cpp -o pyx/ANNarchyCython_3.x.o -L. -I. -Ibuild -fopenmp -std=c++0x -fpermissive 

build/%(obj_type)s : build/%(src_type)s
\tg++ %(flag)s -fPIC -pipe -fpermissive -std=c++0x -fopenmp -I. -c $< -o $@

ANNarchyCPP : $(OBJ)
\tg++ %(flag)s -fPIC -shared -fpermissive -std=c++0x -fopenmp -I. build/*.o -o libANNarchyCPP.so

clean:
\trm -rf build/*.o
\trm -rf pyx/*.o
\trm -rf ANNarchyCython.so
"""