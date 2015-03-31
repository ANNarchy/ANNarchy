""" 

    Scalability.py
    
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
from copy import deepcopy
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui

from DataLog import DataLog
from Custom import IntAxis

class Scalability:
    """
    """
    def __init__(self, par_names, par_scale, thread_scale, num_samples):
        """
        """
        self._thread_scale = thread_scale
        self._par_scale = {}
        for i in range(len(par_scale)):
            self._par_scale.update({ par_scale[i]: i })
        
        self._data = {}
        for name in par_names:
            self._data[name] = DataLog(self._par_scale, num_samples, name)
        
    def add_data_set(self, name, par, data):
        """
        """
        for i in xrange(len(data)): 
            self._data[name][par, i] = data[i]

    def analyze_data(self, debug = False):
        self._speedup = {}
        self._efficiency = {}
        
        for name, data in self._data.iteritems():
        
            tmp = deepcopy(data.raw_data())
            for c in range(tmp.shape[1]):
                tmp[:,c] = (tmp[0,c]/tmp[:,c])  # sequential / parallel = speedup 
            
            self._speedup[name] = tmp
       
            tmp2 = deepcopy(tmp)
            for r in range(tmp2.shape[0]):
                tmp2[r,:] = (tmp2[r,:]/float(self._thread_scale[r]))  # parallel / used cores = efficiency 
            
            self._efficiency[name] = tmp2
            
        if debug:
            for name, data in self._data.iteritems():
                Global._print(name+'(raw):')
                Global._print(data.raw_data())
                
                Global._print(name+'(speedup):')
                Global._print(self._speedup[name])
    
                Global._print(name+'(efficiency):')
                Global._print(self._efficiency[name])
            
    def visualize_data(self):
        """
        """
        self._perf_win = pg.GraphicsWindow(title="Performance evaluation")
        col_array = ['r','g','b','c','w']

        for name in self._data.keys():

            #
            #    computation time
            #
            col_iter = iter(col_array)

            plt_data = self._data[name].raw_data()
            x_scale = [i for i in xrange(plt_data.shape[1])]
            
            tmp_plot = self._perf_win.addPlot(title = "sum (measured time)", axisItems = {'bottom': IntAxis('bottom') })
            tmp_plot.addLegend()
            tmp_plot.setLabel('left', "computation time", units='s')
            tmp_plot.setLabel('bottom', "number of connections",)
            
            for i in xrange( plt_data.shape[0] ):
                tmp_plot.plot(x_scale, 
                              plt_data[i,:], 
                              pen = next(col_iter), 
                              name = str(self._thread_scale[i])+' thread(s)')

            #
            #    speedup
            #
            col_iter = iter(col_array)

            plt_data = self._speedup[name]
            x_scale = [i for i in xrange(plt_data.shape[1])]
            
            tmp_plot2 = self._perf_win.addPlot(title = "sum (speedup)", axisItems = {'bottom': IntAxis('bottom') })
            tmp_plot2.addLegend()
            tmp_plot2.setLabel('left', "achieved speedup")
            tmp_plot2.setLabel('bottom', "number of connections",)
            
            for i in xrange( plt_data.shape[0] ):
                tmp_plot2.plot(x_scale, 
                              plt_data[i,:], 
                              pen = next(col_iter), 
                              name = str(self._thread_scale[i])+' thread(s)')
                
            #
            #    efficiency
            #
            col_iter = iter(col_array)

            tmp_plot3 = self._perf_win.addPlot(title = "sum (efficiency)", axisItems = {'bottom': IntAxis('bottom') })
            tmp_plot3.addLegend()
            tmp_plot3.setLabel('left', "achieved efficiency")
            tmp_plot3.setLabel('bottom', "number of connections",)

            plt_data = self._efficiency[name]
            
            for i in xrange( plt_data.shape[0] ):
                tmp_plot3.plot(x_scale, 
                               plt_data[i,:], 
                               pen = next(col_iter), 
                               name = str(self._thread_scale[i])+' thread(s)')

    def save_as_mat(self):

        save_data = {}

        for name, data in self._data.iteritems():
            save_data['parameter'] = [ x for x in self._par_scale.keys() ]
            save_data['threads'] = self._thread_scale
            save_data[name] = data.raw_data()
            save_data[name+'_speedup'] = self._speedup[name]
            save_data[name+'_efficiency'] = self._efficiency[name]

        from scipy.io import savemat
        savemat('scalability.mat', save_data)