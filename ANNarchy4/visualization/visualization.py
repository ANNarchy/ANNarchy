"""
* visualization.py
*
* Possible improvements:
*
*   -   configurator of visualization fields, to create the visualization plot dictionary currently 
*       provided as plot_data to Visualization
"""
import matplotlib
from pkg_resources import parse_version

enable_vis = True
if parse_version(matplotlib.__version__) < parse_version('1.0'):
    enable_vis = False
    print 'Insufficient matplotlib version.'

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from math import ceil, sqrt
from datetime import datetime 

class Render:
    """
    base class for all render objects.
    """
    def __init__(self, handle, plot_data):
        """
        """
        self.handle = handle
        if 'min' in plot_data.keys():
            self.p_min = plot_data['min']
        else:
            self.p_min = 0.0

        if 'max' in plot_data.keys():
            self.p_max = plot_data['max']
        else:
            self.p_max = 1.0

        if 'cmap' in plot_data.keys():
            try:
                self.p_map =  eval('cm.'+plot_data['cmap'])
            except:
                print 'unknown color map'
                self.p_map = cm.hot
        else:
            self.p_map = cm.hot
            
        self.title = plot_data['title']
            
    def update(self):
        """
        * Update current plot.
        *
        * Implemented by derived classes.
        """        
        print 'It is recommended to use derived objects.'
        
class Visualization:
    """
    Visualization class. 
    Later extended by complete config or a GUI to configure.
    """
    def __init__(self, data):
        """
        Constructor
        
        Parameter:
        * data      list of dictionary describing the single plots.
        """
        if not enable_vis:
            return 
    
        #
        # currently we assume a squared order of visualization fields
        num_row = int(ceil(sqrt(len(data))))
        num_col = num_row
        
        self.time = 0
        
        self.figure, self.handles = plt.subplots(num_row, num_col)
        self.plots = []

        i = 0 # plot id
        for y in xrange(num_row):
            for x in xrange(num_col):
                    
                if i >= len(data):
                    self.figure.delaxes(self.handles[y][x])
                    continue # no data
                                        
                if 'pop' in data[i].keys():
                    if not 'title' in data[i].keys():
                        data[i].update( { 'title': eval("(data[i]['pop']).name") } )

                    if len(data)>1:
                        self.plots.append(VisualizePopulationVar(self.handles[y][x], data[i]))
                    else:
                        self.plots.append(VisualizePopulationVar(self.handles, data[i]))
                    i += 1
                elif 'proj' in data[i].keys():
                    if not 'title' in data[i].keys():
                        data[i].update( { 'title': 'Projection' } )

                    if len(data)>1:                    
                        self.plots.append(VisualizeProjectionVar(self.handles[y][x], data[i]))
                    else:
                        self.plots.append(VisualizeProjectionVar(self.handles, data[i]))
                    i += 1
                else:
                    print 'Incorrect plot data.'
                
        self.figure.canvas.draw()
        
        plt.ion()   #interactive mode
        plt.show()  
        
    def render(self, interval=1, show_time=False):
        """
        Render all plots.
        
        Parameters:
        
        * interval      all plots are updated each #interval steps.
        * show_time     how long the visualition
        """
        if not enable_vis:
            return 

        t_start = datetime.now()
        if interval == 1:            
            for plot in self.plots:
                plot.update()
            self.figure.canvas.draw()
            plt.draw()

        elif self.time % interval == 0:
            for plot in self.plots:
                plot.update()
            self.figure.canvas.draw()
            plt.draw()

        t_stop = datetime.now()
        if show_time:
            print 'Visualization:\t', t_stop-t_start
            
        self.time += 1
        
class VisualizePopulationVar(Render):
    """
    Render a population varibale to the given plot.
    """
    def __init__(self, handle, plot_data):
        """
        Constructor
        
        handle    handle of the plot.
        init_data dictionary contain all plot data
        """
        Render.__init__(self, handle, plot_data)
        self.pop = plot_data['pop']
        self.var = plot_data['var']
        
        self.cmd ='self.pop.get_variable(self.var)'
        self.handle.set_title(self.title)
        self.image = self.handle.imshow(eval(self.cmd), 
                                        cmap= self.p_map, 
                                        interpolation='nearest', 
                                        vmin= self.p_min, 
                                        vmax = self.p_max)
        
        
    def update(self):
        """
        Update current plot.
        """
        self.image.set_data(eval(self.cmd))        
        
class VisualizeProjectionVar(Render):
    """
    Render a projection varibale to the given plot.
    """
    def __init__(self, handle, plot_data):
        """
        Constructor
        
        handle    handle of the plot.
        init_data dictionary contain all plot data
        """
        Render.__init__(self, handle, plot_data)
        
        self.proj = plot_data['proj']
        self.var = plot_data['var']
        self.ranks = []
        if 'ranks' in plot_data.keys():
            self.ranks = plot_data['ranks']
        
        self.cmd = 'self.proj.gather_data(\''+self.var+'\')'
        self.handle.set_title(self.title)
        self.image = self.handle.imshow(eval(self.cmd), 
                                        cmap= self.p_map, 
                                        interpolation='nearest', 
                                        vmin= self.p_min, 
                                        vmax = self.p_max)
        
    def update(self):
        """
        Update current plot.
        """
        self.image.set_data(eval(self.cmd))        

