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
import numpy as np

class DataLog(object):
    """
    Class storing recorded data in two dimensional fashion
    
        scale * amount of measurement points
    """
    def __init__(self, data_pts, meas_pts, operation):
        """
        Constructor

        * Parameters *:
        
            * *data_pts*: which data points are recorded ( threads, neuron amounts, connection amounts)
            * *meas_pts*: amount of sample points should be recorded per data point.
            * *operation*: the name of the operation for easier identification of data logs.
            
        * Example *:
        
            >>> DataLog( [1,2,4], 500, 'sum' )
            
            creates a data log 'sum' for the data pts: 1, 2, 4 and reserve 500 storage points.
        """
        self._operation = operation
        self._meas_pts = meas_pts
        self._data_pts = data_pts            
        self._data = np.zeros((meas_pts, len(data_pts))) 

        self._mean = np.array([ 0.0 for x in xrange(len(self._data_pts))])
        self._std = np.array([ 0.0 for x in xrange(len(self._data_pts))])
        self._min = np.array([ 0.0 for x in xrange(len(self._data_pts))])
        self._max = np.array([ 0.0 for x in xrange(len(self._data_pts))])
        
    def __getitem__(self, idx):
        if not isinstance(idx, tuple) or len(idx) != 2:
            raise IndexError
        
        return self._data[ idx[1], self._data_pts[idx[0]] ]

    def __setitem__(self, idx, value):
        if not isinstance(idx, tuple) or len(idx) != 2:
            raise IndexError

        self._data[ idx[1], self._data_pts[idx[0]] ] = value

    def analyse_data(self):
        """
        Calculate mean and standard deviation for logged data.
        """
        for t in xrange(len(self._data_pts)):
            self._mean[t] = np.mean(self._data[:,t], axis=0)
            self._std[t] = np.std(self._data[:,t], axis=0)
            self._min[t] = np.amin(self._data[:,t], axis=0)
            self._max[t] = np.amax(self._data[:,t], axis=0)

    def mean(self):
        """
        """
        mean = np.zeros(len(self._data_pts))
        
        i = 0
        for t in self._data_pts.keys(): # assume here, that threads are stored ascending in the key value pair
            mean[i] = self._mean[self._data_pts[t]]
            i+=1
        
        return mean

    def variance(self):
        """
        implementation of the corrected variance
        """
        variance = np.zeros(len(self._data_pts))
        
        i = 0
        for t in self._data_pts.keys(): # assume here, that threads are stored ascending in the key value pair
            var = 0.0;
            x_mean = self.mean(self._data_pts[t])
            for x in self._data_pts[t]:
                var += (x - x_mean) ** 2
            
            variance[i] = var / (len(self._data_pts[t]) - 1 )
            i+=1
        
        return variance

    def min(self):
        """
        """
        min = np.zeros(len(self._data_pts))
        
        i = 0
        for t in self._data_pts.keys(): # assume here, that threads are stored ascending in the key value pair
            min[i] = self._min[self._data_pts[t]]
            i+=1
        
        return min
    
    def max(self):
        """
        """
        max = np.zeros(len(self._data_pts))
        
        i = 0
        for t in self._data_pts.keys(): # assume here, that threads are stored ascending in the key value pair
            max[i] = self._max[self._data_pts[t]]
            i+=1
        
        return max

    def std(self):
        """
        """
        std = np.zeros(len(self._data_pts))
        
        i = 0
        for t in self._data_pts.keys(): # assume here, that threads are stored ascending in the key value pair
            std[i] = self._std[self._data_pts[t]]
            i+=1
        
        return std
    
    def asc_idx(self):
        """
        """
        return np.array([i+1 for i in xrange(len(self._data_pts))])
        
    def raw_data(self):
        return self._data
    
    def save_to_file(self, name):
        """
        Save the data to file *name*.
        """
        np.savetxt(name, self._data, delimiter=',')
