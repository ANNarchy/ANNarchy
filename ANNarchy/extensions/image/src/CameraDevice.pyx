# cython: embedsignature=True

from libcpp.vector cimport vector
from libcpp cimport bool

__version__ = '0.01'

cdef extern from "CameraDeviceCPP.h":

    cdef cppclass CameraDeviceCPP:
        
        CameraDeviceCPP(int, int, int, int)
        vector[float] GrabImage()
        bool isOpened()
        void open(int)
        void release()
  
cdef class CameraDevice:
    
    cdef CameraDeviceCPP* camera_
    cdef vector[float] data

    def __cinit__(self, int id, int width, int height, int depth):
        self.camera_ = new CameraDeviceCPP(id, width, height, depth)

    def grab_image(self):
        self.data = self.camera_.GrabImage()
        return self.data
		
    def is_opened(self):
        return self.camera_.isOpened()
		
    def open(self, int id):
        self.camera_.open(id)
	    
    def release(self):
        self.camera_.release()
