# cython: embedsignature=True

"""

    Network.pyx

    This file is part of ANNarchy.
    
    Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
    Helge Ülo Dinkelbach <helge.dinkelbach@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ANNarchy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
from libcpp.vector cimport vector
cdef extern from "../build/Network.h":
    cdef cppclass Network:
        Network()
        
        void destroy()

        int getTime()

        void setTime(int)

        void setNumThreads(int)
        
        void run(int nbSteps)

        int run_until(int nbSteps, vector[int] populations)
		
cdef extern from "../build/Network.h" namespace "Network":
    cdef Network* instance()

cdef class pyNetwork:

    cdef Network* cInstance

    def __cinit__(self):
        self.cInstance = instance()

    def destroy(self):
        self.cInstance.destroy()

    def get_time(self):
        return self.cInstance.getTime()

    def set_time(self, time):
        self.cInstance.setTime(time)

    def set_num_threads(self, threads):
        self.cInstance.setNumThreads(int(threads))
        
    def run(self, int nbSteps):
        self.cInstance.run(nbSteps)
        
    def run_until(self, int nbSteps, list pops):
        return self.cInstance.run_until(nbSteps, pops)
