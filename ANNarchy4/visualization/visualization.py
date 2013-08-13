"""
visualization.py
"""
import matplotlib
if matplotlib.__version__ < 1.0:
    print 'Insufficient matplotlib version.'
    exit(0)

import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
        num_row = 1
        num_col = len(data)
        
        self.time = 0
        
        self.figure, self.handles = plt.subplots(num_row, num_col)
        self.plots = []
        
        for i in xrange(num_col):
            #if isinstance(data[i]['pop'], )
            self.plots.append(VisualizePopulationVar(self.handles[i], data[i]))
        
        self.figure.canvas.draw()
        
        plt.ion()   #interactive mode
        plt.show()  
        
    def render(self, interval):
        """
        * Render all plots.
        *
        * interval      all plots are updated each #interval steps.
        """
        if self.time % interval == 0:
            self.figure.canvas.draw()
            plt.draw()
            
        else:
            for plot in self.plots:
                plot.update()
        
        self.time += 1
        
class VisualizePopulationVar(Render):
    """
    Render a population varibale to the given plot.
    """

    def __init__(self, handle, plot_data):
        """
        * Constructor
        *
        * handle    handle of the plot.
        * init_data dictionary contain all plot data
        """
        Render.__init__(self, handle, plot_data)
        
        self.pop = plot_data['pop']
        self.var = plot_data['var']
        
        # variable access
        # TODO: maybe a better solution?
        self.cmd = 'self.pop.cyInstance.'+self.var+'.reshape(20,20)'
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