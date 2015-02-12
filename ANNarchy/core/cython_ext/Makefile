#
#	Makefile for cythonize the ANNarchy extensions 'Connector', 'Coordinates' and 'Transformations'
#
#	11.02.2015 (hd)
#
#	Used only on darwin-based platforms, as the distutils on python 2.x does not work properly there.
#	Normally these steps are done as part of the distutils.setup(). The used compiler flags are extracted from non-working cythonize attempt.
#
all:

	cython --cplus Connector.pyx
	clang++ -fno-strict-aliasing -fno-common -dynamic -g -Os -pipe -DENABLE_DTRACE -DMACOSX -DNDEBUG -Wshorten-64-to-32 -fwrapv -Wall -Wstrict-prototypes -I/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include -I/System/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7 --shared Connector.cpp -lpython -o Connector.so

	cython --cplus Coordinates.pyx
	clang++ -fno-strict-aliasing -fno-common -dynamic -g -Os -pipe -DENABLE_DTRACE -DMACOSX -DNDEBUG -Wshorten-64-to-32 -fwrapv -Wall -Wstrict-prototypes -I/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include -I/System/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7 --shared Coordinates.cpp -lpython -o Coordinates.so

	cython --cplus Transformations.pyx
	clang++ -fno-strict-aliasing -fno-common -dynamic -g -Os -pipe -DENABLE_DTRACE -DMACOSX -DNDEBUG -Wshorten-64-to-32 -fwrapv -Wall -Wstrict-prototypes -I/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include -I/System/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7 --shared Transformations.cpp -lpython -o Transformations.so

clean:
	rm -rf *.cpp	# results of 1st step
	rm -rf *.so		# results of 2nd step