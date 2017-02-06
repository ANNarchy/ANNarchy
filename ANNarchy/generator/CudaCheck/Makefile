all: cuda_check.so

cuda_check_cu.o:
	nvcc -c cuda_check.cu -Xcompiler -fPIC -o cuda_check_cu.o

cuda_check.cpp:
	cython --cplus cuda_check.pyx

cuda_check.so: cuda_check_cu.o cuda_check.cpp
	g++ cuda_check.cpp -fPIC -shared -g -I. `/usr/bin/python2-config --includes` cuda_check_cu.o -lcudart -o cuda_check.so -L/usr/lib `/usr/bin/python2-config --ldflags --libs`

clean:
	rm -f cuda_check_cu.o
	rm -f cuda_check.cpp
	rm -f cuda_check.so
