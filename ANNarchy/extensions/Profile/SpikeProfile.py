""" 

    SpikeProfile.py
    
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
import datetime
import pyqtgraph as pg
from math import ceil, floor, sqrt

from ANNarchy.core import Global
from ANNarchy import *

from .DataLog import DataLog
from .Custom import IntAxis
from .Profile import Profile

class SpikeProfile(Profile):
    """
    Measuring and analyzing of rate-coded populations.
    """                
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

        for name, data in self._pop_data.items():
            data['cond'][thread, trial] = self._average_conductance(name, begin, end)
            data['del'][thread, trial] = self._average_spike_delivery(name, begin, end)
            data['pre'][thread, trial] = self._average_pre_event(name, begin, end)
            data['post'][thread, trial] = self._average_post_event(name, begin, end)
    
    def measure_func(self, func, steps):
        """
        Profiles the provided simulation loop
        """
        Global._print(self._threads)
        for thread in self._threads:
            Global._network['instance'].set_num_threads(thread)
        
            for trial in xrange(self._num_trials):
                func(steps)
                
                self.measure(thread, trial, trial*steps, (trial+1)*steps)
                
            self.reset_timer()
               
    def analyse_data(self):
        """
        Iterate over all data fields and create evaluation data.
        """
        # evaluate datasets - network
        self._net_data.analyse_data()
        
        #evaluate datasets - layer and operation wise
        for pop in self._pop_data.itervalues():
            for tmp in pop.itervalues():
                tmp.analyse_data()

    def print_data(self):
        Global._print('overall:')
        Global._print('    mean:', self._net_data.mean())
        Global._print('    min: ', self._net_data.min())
        Global._print('    max: ', self._net_data.max())

        for name, data in self._pop_data.items():
            Global._print(name,'(conductance):')
             
            Global._print('    mean:', data['cond'].mean())
            Global._print('    min: ', data['cond'].min())
            Global._print('    max: ', data['cond'].max())

            Global._print(name,'(spike delivery):')
             
            Global._print('    mean:', data['del'].mean())
            Global._print('    min: ', data['del'].min())
            Global._print('    max: ', data['del'].max())

            Global._print(name,'(pre event eval):')
             
            Global._print('    mean:', data['pre'].mean())
            Global._print('    min: ', data['pre'].min())
            Global._print('    max: ', data['pre'].max())

            Global._print(name,'(post event eval):')
             
            Global._print('    mean:', data['post'].mean())
            Global._print('    min: ', data['post'].min())
            Global._print('    max: ', data['post'].max())
        
    def visualize_data(self, error_bar = False):
        """
        Visualize current analyzed data with pyqtgraph.
        
        Parameter:
        
        * *error_bar*: show the min and max values for all data sets (default = False)
        """
        self._net_win = pg.GraphicsWindow(title="Speedup: network overall")
        # additional customizations        
        #self._net_win.setBackground('w')
        self._net_win.resize(1000,600)
        col_array = ['r','g','b','c','w']
        
        x_scale = np.array([i for i in xrange(len(self._threads))])
        for k,v in self._threads.items():
            x_scale[v] = k

                
        p1 = self._net_win.addPlot(title = "")
        p1.setLabel('left', "computation time", units='ms')
        p1.setLabel('bottom', "number of trials",)
        p1.plot(x_scale, self._net_data._mean)
        # additional customizations        
        #p1.getAxis('bottom').setPen('r')

        col_iter = iter(col_array)
 
        self._pop_win1 = []
        self._pop_win2 = []
 
        pop_mean_label = { 'left' : "computation time", 'bottom': "number of threads" }
         
        def create_error_bar(idx, mean, min, max, std):
            """
            for equal configuration on all plots
            """
            err = pg.ErrorBarItem( x=idx, 
                                   y=mean,
                                   top=std, 
                                   bottom=std, 
                                   beam=0.5)
            return err

        #
        # plot the population data
        for name, data in self._pop_data.items():
            col_iter2 = iter(col_array)
             
            tmp = pg.GraphicsWindow(title="raw data: "+name)
            tmp.resize(1000,600)
            tmp2 = pg.GraphicsWindow(title="Evaluation: "+name)
            tmp2.resize(1000,600)
 
            #=============================#
            #     conductance             #
            #=============================#
            #
            # raw data
            plt_data = data['cond']._data
            x_scale = [i for i in xrange(plt_data.shape[0])]
            thread_num = np.array([i for i in xrange(len(self._threads))])
            for k,v in self._threads.items():
                thread_num[v] = k
                          
            tmp_plot = tmp.addPlot(title = "conductance", axisItems = {'bottom': IntAxis('bottom') })
            tmp_plot.addLegend()
            tmp_plot.setLabel('left', "computation time", units='s')
            tmp_plot.setLabel('bottom', "number of trials",)
             
            col_iter = iter(col_array)
            for i in xrange( plt_data.shape[1] ):
                tmp_plot.plot(x_scale, plt_data[:,i], pen = next(col_iter), name = str(thread_num[i])+' thread(s)' )
 
            #
            # mean, min, max
            pop_mean_plot = tmp2.addPlot(title = "conductance", axisItems = {'bottom': IntAxis('bottom') })
            pop_mean_plot.setLabel('left', "computation time", units='s')
            pop_mean_plot.setLabel('bottom', "number of cores",)
 
            if error_bar:
                err = create_error_bar(data['cond'].asc_idx(), data['cond'].mean(), data['cond'].min(), data['cond'].max(), data['cond'].std())
                pop_mean_plot.addItem(err)
            pop_mean_plot.plot( thread_num, 
                                data['cond'].mean(), 
                                pen = { 'color':next(col_iter2), 'width': 2 }, 
                                labels=pop_mean_label )
 
            #=============================#
            #     spike delivery          #
            #=============================#
            #
            # raw data
            plt_data = data['del']._data
            tmp_plot = tmp.addPlot(title = "spike delivery", axisItems = {'bottom': IntAxis('bottom') })
            tmp_plot.addLegend()
            tmp_plot.setLabel('left', "computation time", units='s')
            tmp_plot.setLabel('bottom', "number of trials",)
 
            col_iter = iter(col_array)
            for i in xrange( plt_data.shape[1] ):
                tmp_plot.plot(x_scale, plt_data[:,i], pen = next(col_iter), name = str(thread_num[i])+' thread(s)' )
 
            #
            # mean, min, max
            pop_mean_plot = tmp2.addPlot(title = "spike delivery", axisItems = {'bottom': IntAxis('bottom') })
            pop_mean_plot.setLabel('left', "computation time", units='s')
            pop_mean_plot.setLabel('bottom', "number of cores",)
            if error_bar:
                err = create_error_bar(data['del'].asc_idx(), data['del'].mean(), data['del'].min(), data['del'].max(), data['del'].std())
                pop_mean_plot.addItem(err)
            pop_mean_plot.plot( thread_num, 
                                data['del'].mean(), 
                                pen = { 'color':next(col_iter2), 'width': 2 }, 
                                labels=pop_mean_label )
 
            #
            # first plot row completed
            tmp.nextRow()
            tmp2.nextRow()
 
            #=============================#
            #     pre event               #
            #=============================#
            #
            # raw data
            plt_data = data['pre']._data
            tmp_plot = tmp.addPlot(title = "pre_event", axisItems = {'bottom': IntAxis('bottom') })
            tmp_plot.addLegend()
            tmp_plot.setLabel('left', "computation time", units='s')
            tmp_plot.setLabel('bottom', "number of trials",)
             
            col_iter = iter(col_array)
            for i in xrange( plt_data.shape[1] ):
                tmp_plot.plot(x_scale, plt_data[:,i], pen = next(col_iter), name = str(thread_num[i])+' thread(s)' )
             
            #
            # mean, min, max
            pop_mean_plot = tmp2.addPlot(title = "pre_event", axisItems = {'bottom': IntAxis('bottom') })
            pop_mean_plot.setLabel('left', "computation time", units='s')
            pop_mean_plot.setLabel('bottom', "number of cores",)
            if error_bar:
                err = create_error_bar(data['pre'].asc_idx(), data['pre'].mean(), data['pre'].min(), data['pre'].max(), data['pre'].std())
                pop_mean_plot.addItem(err)
            pop_mean_plot.plot( thread_num, 
                                data['pre'].mean(), 
                                pen = { 'color':next(col_iter2), 'width': 2 }, 
                                labels=pop_mean_label )
             
            #=============================#
            #     post event              #
            #=============================#
            #
            # raw data
            plt_data = data['post']._data
            tmp_plot = tmp.addPlot(title = "post event", axisItems = {'bottom': IntAxis('bottom') })
            tmp_plot.addLegend()
            tmp_plot.setLabel('left', "computation time", units='s')
            tmp_plot.setLabel('bottom', "number of trials",)
 
            col_iter = iter(col_array)
            for i in xrange( plt_data.shape[1] ):
                tmp_plot.plot(x_scale, plt_data[:,i], pen = next(col_iter), name = str(thread_num[i])+' thread(s)' )
 
            #
            # mean, min, max
            pop_mean_plot = tmp2.addPlot(title = "post event", axisItems = {'bottom': IntAxis('bottom') })
            pop_mean_plot.setLabel('left', "computation time", units='s')
            pop_mean_plot.setLabel('bottom', "number of cores",)
            if error_bar:
                err = create_error_bar(data['post'].asc_idx(), data['post'].mean(), data['post'].min(), data['post'].max(), data['post'].std())
                pop_mean_plot.addItem(err)
            pop_mean_plot.plot( thread_num, 
                                data['post'].mean(), 
                                pen = { 'color':next(col_iter2), 'width': 2 }, 
                                labels=pop_mean_label )
             
            self._pop_win1.append(tmp)
            self._pop_win2.append(tmp2)
            
    def save_to_file(self):
        """
        Save the recorded data to several files.
        """
        pass
        #=======================================================================
        # time = datetime.datetime.now().strftime("%Y%m%d_%H-%M")
        # 
        # if self._net_data:
        #     if self._name == None:
        #         out_file = self._folder+'/'+time+'_profile_overall.csv'
        #     else:
        #         out_file = self._folder+'/'+self._name+'_overall.csv'
        #     self._net_data.save_to_file(out_file)
        #     
        # empty_row = np.zeros((self._num_trials,1))
        # for name, data in self._pop_data.items():
        #     if self._name == None:
        #         out_file = self._folder+'/'+time+'_profile_name.csv'
        #     else:
        #         out_file = self._folder+'/'+self._name+'.csv'
        #     
        #     complete = np.concatenate( 
        #                     ( data['sum']._data, empty_row,
        #                       data['step']._data, empty_row,
        #                       data['local']._data, empty_row,
        #                       data['global']._data, empty_row
        #                     ), axis = 1
        #                 )
        #     
        #     np.savetxt(out_file, complete, delimiter=',')        
        #=======================================================================
        
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

    def _last_step_conductance(self, name):
        if isinstance(name, str):
            return self._profile_instance.lastTimeConductance(name)
        
    def _average_conductance(self, name, begin, end):
        if isinstance(name, str):
            return self._profile_instance.avgTimeConductance(name, begin, end)

    def _last_step_spike_delivery(self, name):
        if isinstance(name, str):
            return self._profile_instance.lastTimeSpikeDelivery(name)
        
    def _average_spike_delivery(self, name, begin, end):
        if isinstance(name, str):
            return self._profile_instance.avgTimeSpikeDelivery(name, begin, end)

    def _last_step_pre_event(self, name):
        if isinstance(name, str):
            return self._profile_instance.lastTimePreEvent(name)
        
    def _average_pre_event(self, name, begin, end):
        if isinstance(name, str):
            return self._profile_instance.avgTimePreEvent(name, begin, end)

    def _last_step_post_event(self, name):
        if isinstance(name, str):
            return self._profile_instance.lastTimePostEvent(name)
        
    def _average_post_event(self, name, begin, end):
        if isinstance(name, str):
            return self._profile_instance.avgTimePostEvent(name, begin, end)
        