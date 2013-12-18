# cython: embedsignature=True

"""

    Profile.pyx

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
from libcpp.string cimport string

cdef extern from "../build/Profile.h":
    cdef cppclass Profile:
        Profile()

        float getAvgTimeSum(string name, int begin, int end)
        
        float lastRecordedTimeSum(string name)

        float getAvgTimeStep(string name, int begin, int end)
        
        float lastRecordedTimeStep(string name)

        float getAvgTimeLocal(string name, int begin, int end)
        
        float lastRecordedTimeLocal(string name)

        float getAvgTimeGlobal(string name, int begin, int end)
        
        float lastRecordedTimeGlobal(string name)
        
cdef extern from "../build/Profile.h" namespace "Profile":
    cdef Profile* profileInstance()

cdef class pyProfile:

    cdef Profile* cInstance

    def __cinit__(self):
        self.cInstance = profileInstance()

    def lastTimeSum(self, name):
        return self.cInstance.lastRecordedTimeSum(name)
    
    def avgTimeSum(self, name, begin, end):
        return self.cInstance.getAvgTimeSum(name, begin, end)

    def lastTimeStep(self, name):
        return self.cInstance.lastRecordedTimeStep(name)
    
    def avgTimeStep(self, name, begin, end):
        return self.cInstance.getAvgTimeStep(name, begin, end)

    def lastTimeLocal(self, name):
        return self.cInstance.lastRecordedTimeLocal(name)
    
    def avgTimeLocal(self, name, begin, end):
        return self.cInstance.getAvgTimeLocal(name, begin, end)

    def lastTimeGlobal(self, name):
        return self.cInstance.lastRecordedTimeGlobal(name)
    
    def avgTimeGlobal(self, name, begin, end):
        return self.cInstance.getAvgTimeGlobal(name, begin, end)
