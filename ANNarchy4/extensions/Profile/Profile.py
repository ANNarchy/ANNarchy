""" 

    Profile.py
    
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
import exceptions
from ANNarchy4 import *
import numpy

import datetime
import matplotlib.pyplot as plt

import pyqtgraph as pg
from math import floor, ceil, sqrt

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
            idx = numpy.linspace(minVal, maxVal, num=11)
            if int(floor(idx[1])) > 0:
                return [(int(floor(idx[1])),0)]
            else:
                return super.tickSpacing(self,minVal, maxVal, size)  
            
class DataLog(object):
    def __init__(self, threads, num_trials):
        """
        Constructor
        """
        self._num_trials = threads
        self._threads = threads            
        self._data = numpy.zeros((num_trials, len(threads))) 

        self._mean = [ 0.0 for x in xrange(len(self._threads))]
        self._std = [ 0.0 for x in xrange(len(self._threads))]
        
    def __getitem__(self, idx):
        if not isinstance(idx, tuple) or len(idx) != 2:
            raise IndexError
        
        return self._data[ idx[1], self._threads[idx[0]] ]

    def __setitem__(self, idx, value):
        if not isinstance(idx, tuple) or len(idx) != 2:
            raise IndexError

        self._data[ idx[1], self._threads[idx[0]] ] = value

    def analyse_data(self):
        """
        Calculate mean and standard deviation for logged data.
        """
        for t in xrange(len(self._threads)):
            self._mean[t] = numpy.mean(self._data[:,t], axis=0)
            self._std[t] = numpy.std(self._data[:,t], axis=0)
        
    def save_to_file(self, name):
        """
        Save the data to file *name*.
        """
        numpy.savetxt(name, self._data, delimiter=',')
                
class Profile:
    def __init__(self, num_threads, num_trials, name='profile'):
        """
        Constructor, setup the overall configuration of profiling session.
        
        Parameter:
        
            * *num_threads*: thread configuration, array consisting of thread amounts of the different runs.
            * *num_trials*: amount of measurement points for each thread configuration
        """
        try:
            import ANNarchyCython
        except exceptions.ImportError:
            print 'Error on Profile'
        else:
            print 'Inited profiler.'
            self._profile_instance = ANNarchyCython.pyProfile()
            self._network = ANNarchyCython.pyNetwork()
            
        self._name = name
        self._threads = {}
        for i in range(len(num_threads)):
            self._threads.update({ num_threads[i]: i })

        self._num_trials = num_trials
            
        self._net_data = None
        self._pop_data = {}

    def add_to_profile(self, object):
        """
        Which network objects should be tracked.
        
        Parameter:
        
            * *object*: either 'network' or a population name
        """
        if object == "network":
            self._net_data = DataLog(self._threads, self._num_trials)
        else:
            self._pop_data[object] = { 'sum' : DataLog(self._threads, self._num_trials),
                                       'step' : DataLog(self._threads, self._num_trials), 
                                       'local' : DataLog(self._threads, self._num_trials),
                                       'global' : DataLog(self._threads, self._num_trials)
                                     }
        
    def measure(self, thread, trial, begin, end):
        """
        Retrieve measure data.
        
        In general for every time step the corresponding times are taken. Since the user can not
        interact with the profiler, when simulate() runs, he can retrieve the average over the last
        time steps. So normally the difference *end* - *begin* will be exactly the simulation duration. 
        
        Parameters:
        
            * *thread*: thread amount
            * *trial*: measure point
            * *begin*: begin of measure (offset to last reset)
            * *end*: end of measure (offset to last reset)
            
        """
        if self._net_data:
            self._net_data[thread, trial] = self._average_net( begin, end )
        
        for name, data in self._pop_data.iteritems():
            data['sum'][thread, trial] = self._average_sum(name, begin, end)
            data['step'][thread, trial] = self._average_step(name, begin, end)
            data['local'][thread, trial] = self._average_local(name, begin, end)
            data['global'][thread, trial] = self._average_global(name, begin, end)
            
    def analyse_data(self):
        
        self._net_win = pg.GraphicsWindow(title="Speedup: network overall")
        self._net_win.resize(1000,600)
        
        x_scale = [i for i in xrange(len(self._threads))]
        for k,v in self._threads.iteritems():
            x_scale[v] = k

        # evaluate datasets - network
        self._net_data.analyse_data()
                
        p1 = self._net_win.addPlot(title = "")
        p1.plot(x_scale, self._net_data._mean)


        # evaluate datasets - layer and operation wise
        for pop in self._pop_data.itervalues():
            for tmp in pop.itervalues():
                tmp.analyse_data()

        self._pop_win = pg.GraphicsWindow(title="Speedup: population / operation wise")
        self._pop_win.resize(1000,600)

        num_row = int(floor(sqrt(len(self._pop_data))))
        num_col = int(ceil(sqrt(len(self._pop_data))))
        
        #
        # plot the population data
        pop_iter = iter(self._pop_data)
        for y in xrange( num_row ):
            for x in xrange( num_col ):
                
                #try:
                it = next(pop_iter)
                pop_mean_plot = self._pop_win.addPlot(title = it, axisItems = {'bottom': IntAxis('bottom') })
                pop_mean_plot.addLegend( offset = (0,50))
                
                mean_data = self._pop_data[it]['sum']._mean
                std_data = self._pop_data[it]['sum']._std
                pop_mean_plot.plot( x_scale, mean_data, pen = ( 255.0, 255.0, 255.0 ), name = 'weighted sum' )
                
                mean_data = self._pop_data[it]['step']._mean
                std_data = self._pop_data[it]['step']._std
                pop_mean_plot.plot( x_scale, mean_data, pen = ( 255.0, 0.0, 0.0 ), name = 'neuron step'  )

                mean_data = self._pop_data[it]['local']._mean
                std_data = self._pop_data[it]['local']._std
                pop_mean_plot.plot( x_scale, mean_data, pen = ( 0.0, 255.0, 0.0 ), name = 'local learn'  )

                mean_data = self._pop_data[it]['global']._mean
                std_data = self._pop_data[it]['global']._std
                pop_mean_plot.plot( x_scale, mean_data, pen = ( 0.0, 0.0, 255.0 ), name = 'global learn'  )
            
            if ( num_row > 1):
                self.pop_win.nextRow()
        
        thread_num = [i for i in xrange(len(self._threads))]
        for k,v in self._threads.iteritems():
            thread_num[v] = k
        self._pop_win2 = []
        for name, data in self._pop_data.iteritems():
        
            tmp = pg.GraphicsWindow(title="Detailed data: "+name)
            tmp.resize(1000,600)

            col_array = ['r','g','b','c','w']
            
            plt_data = data['sum']._data
            x_scale = [i for i in xrange(plt_data.shape[0])]
             
            tmp_plot = tmp.addPlot(title = "sum", axisItems = {'bottom': IntAxis('bottom') })
            tmp_plot.addLegend()
            col_iter = iter(col_array)
            for i in xrange( plt_data.shape[1] ):
                tmp_plot.plot(x_scale, plt_data[:,i], pen = next(col_iter), name = str(thread_num[i])+' thread(s)' )

            plt_data = data['step']._data
            tmp_plot = tmp.addPlot(title = "step", axisItems = {'bottom': IntAxis('bottom') })
            tmp_plot.addLegend()
            col_iter = iter(col_array)
            for i in xrange( plt_data.shape[1] ):
                tmp_plot.plot(x_scale, plt_data[:,i], pen = next(col_iter), name = str(thread_num[i])+' thread(s)' )

            tmp.nextRow()

            plt_data = data['global']._data
            tmp_plot = tmp.addPlot(title = "global", axisItems = {'bottom': IntAxis('bottom') })
            tmp_plot.addLegend()
            col_iter = iter(col_array)
            for i in xrange( plt_data.shape[1] ):
                tmp_plot.plot(x_scale, plt_data[:,i], pen = next(col_iter), name = str(thread_num[i])+' thread(s)' )
            
            plt_data = data['local']._data
            tmp_plot = tmp.addPlot(title = "local", axisItems = {'bottom': IntAxis('bottom') })
            tmp_plot.addLegend()
            col_iter = iter(col_array)
            for i in xrange( plt_data.shape[1] ):
                tmp_plot.plot(x_scale, plt_data[:,i], pen = next(col_iter), name = str(thread_num[i])+' thread(s)' )
            
            self._pop_win2.append(tmp)

    def analyse_data_mp(self):
        """
        """
        num_row = 2
        num_col = 2
        
        #
        # pre setup
        x_scale = [i for i in xrange(len(self._threads))]
        for k,v in self._threads.iteritems():
            x_scale[v] = k
        # evaluate datasets - network
        self._net_data.analyse_data()
        # evaluate datasets - layer and operation wise
        for pop in self._pop_data.itervalues():
            for tmp in pop.itervalues():
                tmp.analyse_data()
                
        #
        #mean and std
        mean_figure, mean_handles = plt.subplots(num_row, num_col)
        plt.suptitle("Mean and STD")
        
        mean_handles[0,0].errorbar(x_scale, self._net_data._mean, yerr=self._net_data._std)
        mean_handles[0,0].set_xlim([1,len(self._threads)])
        mean_handles[0,0].set_xticks(x_scale) 
        
        #
        # population data
        pop_iter = iter(self._pop_data)
        for y in xrange(1,num_row):
            for x in xrange(num_col):
                
                #try:
                it = next(pop_iter)
                
                
                mean_data = self._pop_data[it]['sum']._mean
                std_data = self._pop_data[it]['sum']._std
                p1 = mean_handles[y,x].errorbar(x_scale, mean_data, yerr=std_data)
                mean_handles[y,x].set_xlabel('trials')
                mean_handles[y,x].set_ylabel('time in ms')

                mean_data = self._pop_data[it]['step']._mean
                std_data = self._pop_data[it]['step']._std
                p2 = mean_handles[y,x].errorbar(x_scale, mean_data, yerr=std_data)

                mean_data = self._pop_data[it]['local']._mean
                std_data = self._pop_data[it]['local']._std
                p3 = mean_handles[y,x].errorbar(x_scale, mean_data, yerr=std_data)

                mean_data = self._pop_data[it]['global']._mean
                std_data = self._pop_data[it]['global']._std
                p4 = mean_handles[y,x].errorbar(x_scale, mean_data, yerr=std_data)
                    
                #except:
                #    pass
                
                mean_handles[y,x].set_title(it)
                mean_handles[y,x].legend([p1, p2, p3, p4], ["sum", "step", "local", "global"])
                mean_handles[y,x].set_xlim([1,len(self._threads)])
                mean_handles[y,x].set_xticks(x_scale)
                    
        mean_figure.canvas.draw()

        #
        # raw data
        num_row = 2
        num_col = 2
        for name, data in self._pop_data.iteritems():
            
            raw_figure, raw_handles = plt.subplots(num_row, num_col)
            plt.suptitle(name, fontsize=14)
            
            #
            # population data
            pop_iter = iter(self._pop_data[name])
            for y in xrange(num_row):
                for x in xrange(num_col):
                    
                    try:
                        it = next(pop_iter)
                        plt_data = data[it]._data
                        x_scale = [i for i in xrange(plt_data.shape[0])]
                        
                        for i in xrange( plt_data.shape[1] ):
                            raw_handles[y,x].plot(x_scale, plt_data[:,i])
                        
                        raw_handles[y,x].set_title(it)
                    except:
                        pass

            raw_figure.canvas.draw()
        
        #plt.draw()
        plt.pause(0.01)
            
    def save_to_file(self):
        """
        Save the recorded data to several files.
        """
        time = datetime.datetime.now().strftime("%Y%m%d_%H-%M")
        
        if self._net_data:
            out_file = time+self._name+'_overall.csv'
            self._net_data.save_to_file(out_file)
            
        empty_row = numpy.zeros((self._num_trials,1))
        for name, data in self._pop_data.iteritems():
            out_file = time+self._name+'_'+name+'.csv'
            
            complete = numpy.concatenate( 
                            ( data['sum']._data, empty_row,
                              data['step']._data, empty_row,
                              data['local']._data, empty_row,
                              data['global']._data, empty_row
                            ), axis = 1
                        )
            
            numpy.savetxt(out_file, complete, delimiter=',')        
        
    def reset_timer(self):
        """
        Reset the recorded data.
        """
        self._profile_instance.resetTimer()

    def set_num_threads(self, threads):
        """
        Set the amount of threads used for the next simulate() calls.
        """
        self._network.set_num_threads(threads)

    def _last_step_net(self):
        return self._profile_instance.lastTimeNet()
        
    def _average_net(self, begin, end):
        return self._profile_instance.avgTimeNet( begin, end )

    def _last_step_sum(self, name):
        if isinstance(name, str):
            pop = get_population(name)
            return self._profile_instance.lastTimeSum(pop.class_name)
        
    def _average_sum(self, name, begin, end):
        if isinstance(name, str):
            pop = get_population(name)
            return self._profile_instance.avgTimeSum(pop.class_name, begin, end)

    def _last_step_step(self, name):
        if isinstance(name, str):
            pop = get_population(name)
            return self._profile_instance.lastTimeStep(pop.class_name)
        
    def _average_step(self, name, begin, end):
        if isinstance(name, str):
            pop = get_population(name)
            return self._profile_instance.avgTimeStep(pop.class_name, begin, end)

    def _last_step_local(self, name):
        if isinstance(name, str):
            pop = get_population(name)
            return self._profile_instance.lastTimeLocal(pop.class_name)
        
    def _average_local(self, name, begin, end):
        if isinstance(name, str):
            pop = get_population(name)
            return self._profile_instance.avgTimeLocal(pop.class_name, begin, end)

    def _last_step_global(self, name):
        if isinstance(name, str):
            pop = get_population(name)
            return self._profile_instance.lastTimeGlobal(pop.class_name)
        
    def _average_global(self, name, begin, end):
        if isinstance(name, str):
            pop = get_population(name)
            return self._profile_instance.avgTimeGlobal(pop.class_name, begin, end)
