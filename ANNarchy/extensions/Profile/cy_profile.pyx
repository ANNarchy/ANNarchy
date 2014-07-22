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
from libcpp cimport bool

cdef extern from "../build/Profile.h":
    cdef cppclass Profile:
        Profile()

        void resetTimer()

        double getAvgTimeNet(int begin, int end)
        
        double lastRecordedTimeNet()

        double getAvgTimeSum(string name, int begin, int end, bool)
        
        double getStdDevSum(string name, int begin, int end, bool)
        
        double lastRecordedTimeSum(string name)

        double getAvgTimeStep(string name, int begin, int end, bool)
        
        double getStdDevStep(string name, int begin, int end, bool)
        
        double lastRecordedTimeStep(string name)

        double getAvgTimeLocal(string name, int begin, int end, bool)
        
        double lastRecordedTimeLocal(string name)

        double getAvgTimeGlobal(string name, int begin, int end, bool)
        
        double lastRecordedTimeGlobal(string name)

        double getAvgTimeConductance(string name, int begin, int end)
        
        double lastRecordedTimeConductance(string name)

        double getAvgTimeSpikeDelivery(string name, int begin, int end)
        
        double lastRecordedTimeSpikeDelivery(string name)

        double getAvgTimePreEvent(string name, int begin, int end)
        
        double lastRecordedTimePreEvent(string name)

        double getAvgTimePostEvent(string name, int begin, int end)
        
        double lastRecordedTimePostEvent(string name)
        
cdef extern from "../build/Profile.h" namespace "Profile":
    cdef Profile* profileInstance()

cdef class pyProfile:

    cdef Profile* cInstance

    def __cinit__(self):
        self.cInstance = profileInstance()

    def resetTimer(self):
        self.cInstance.resetTimer()

    def lastTimeNet(self):
        return self.cInstance.lastRecordedTimeNet()
    
    def avgTimeNet(self, begin, end):
        return self.cInstance.getAvgTimeNet( begin, end )

    def lastTimeStep(self, name):
        return self.cInstance.lastRecordedTimeStep(name)

    def lastTimeSum(self, name):
        return self.cInstance.lastRecordedTimeSum(name)
    
    def avgTimeSum(self, name, begin, end, remove_outlier):
        return self.cInstance.getAvgTimeSum(name, begin, end, remove_outlier)

    def stdDevSum(self, name, begin, end, remove_outlier):
        return self.cInstance.getStdDevSum(name, begin, end, remove_outlier)

    def lastTimeStep(self, name):
        return self.cInstance.lastRecordedTimeStep(name)
    
    def avgTimeStep(self, name, begin, end, remove_outlier):
        return self.cInstance.getAvgTimeStep(name, begin, end, remove_outlier)

    def stdDevStep(self, name, begin, end, remove_outlier):
        return self.cInstance.getStdDevStep(name, begin, end, remove_outlier)

    def lastTimeLocal(self, name):
        return self.cInstance.lastRecordedTimeLocal(name)
    
    def avgTimeLocal(self, name, begin, end, remove_outlier):
        return self.cInstance.getAvgTimeLocal(name, begin, end, remove_outlier)

    def lastTimeGlobal(self, name):
        return self.cInstance.lastRecordedTimeGlobal(name)
    
    def avgTimeGlobal(self, name, begin, end, remove_outlier):
        return self.cInstance.getAvgTimeGlobal(name, begin, end, remove_outlier)
    
    def lastTimeConductance(self, name):
        return self.cInstance.lastRecordedTimeConductance(name)
    
    def avgTimeConductance(self, name, begin, end):
        return self.cInstance.getAvgTimeConductance(name, begin, end)

    def lastTimeSpikeDelivery(self, name):
        return self.cInstance.lastRecordedTimeSpikeDelivery(name)
    
    def avgTimeSpikeDelivery(self, name, begin, end):
        return self.cInstance.getAvgTimeSpikeDelivery(name, begin, end)
    
    def lastTimePreEvent(self, name):
        return self.cInstance.lastRecordedTimePreEvent(name)
    
    def avgTimePreEvent(self, name, begin, end):
        return self.cInstance.getAvgTimePreEvent(name, begin, end)
    
    def lastTimePostEvent(self, name):
        return self.cInstance.lastRecordedTimePostEvent(name)
    
    def avgTimePostEvent(self, name, begin, end):
        return self.cInstance.getAvgTimePostEvent(name, begin, end)