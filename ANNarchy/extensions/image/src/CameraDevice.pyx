# cython: embedsignature=True

from libcpp.vector cimport vector
from libcpp cimport bool
import numpy as np
cimport numpy as np
import ANNarchy

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
    cdef pop

    def __cinit__(self, pop, int id, int width, int height, int depth):
        self.pop = pop
        self.camera_ = new CameraDeviceCPP(id, width, height, depth)

    def grab_image(self):
        if hasattr(self.pop, 'cyInstance'):
            setattr(self.pop.cyInstance, 'rate', self.camera_.GrabImage() )
        else:
            self.pop.rate = np.array(self.camera_.GrabImage())
		
    def is_opened(self):
        return self.camera_.isOpened()
		
    def open(self, int id):
        self.camera_.open(id)
	    
    def release(self):
        self.camera_.release()
