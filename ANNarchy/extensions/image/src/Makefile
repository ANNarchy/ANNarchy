
     
all: CameraDevice.o CameraDeviceCPP.o
	g++ -shared -pthread -fPIC -fwrapv -O2 -fno-strict-aliasing *.o -L. -L/usr/lib64 -lpython2.7 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_objdetect -lopencv_video -o CameraDevice.so  

CameraDevice.cpp : CameraDevice.pyx
	cython $< --cplus 
	 
CameraDevice.o : CameraDevice.cpp
	g++ -O2 -pipe -fPIC -Wp,-D_FORTIFY_SOURCE=2 -fexceptions -fstack-protector -D_GNU_SOURCE -fwrapv -I/usr/include/python2.7 -c CameraDevice.cpp -o $@ -L. -I.

CameraDeviceCPP.o : CameraDeviceCPP.cpp
	g++ -O2 -pipe -fPIC -I. -c $< -o $@

clean:
	rm -rf *.o
	rm -rf *.so
	rm CameraDevice.cpp
