# cython: embedsignature=True

"""

    SpikeProjection.pyx

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
cdef extern from "../build/SpikeProjection.h":
    cdef cppclass SpikeProjection:
        SpikeProjection(string pre, string post, int target)
        
cdef class pySpikeProjection:

    cdef SpikeProjection* cInstance

    def __cinit__(self, pre, post, target):
        self.cInstance = new SpikeProjection(pre, post, target)
