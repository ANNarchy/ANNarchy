""" 

    DataLog.py
    
    This file is part of ANNarchy.
    
    Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
    Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

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
import pyqtgraph as pg
import numpy as np
from math import ceil, floor, sqrt

class IntAxis(pg.AxisItem):
    """
    Overridden class of pyqtgraph framework.
    
    To customize the xAxis of the plots ( refer to: customizable plots of the example pyqtgraph package ) 
    """
    def tickSpacing(self, minVal, maxVal, size):
        """
        Parameters as original, returns only major tick spacing of length 1.0
        """
        if maxVal <= 11.0:
            return [(1,0)]
        else:
            idx = np.linspace(minVal, maxVal, num=11)
            if int(floor(idx[1])) > 0:
                return [(int(floor(idx[1])),0)]
            else:
                return pg.AxisItem.tickSpacing(self,minVal, maxVal, size)  
