from ANNarchy.core.Global import get_population, _print
from Custom import *
import numpy

class PerformanceMeasurement:
    
    def __init__(self):
        try:
            import ANNarchyCython
        except exceptions.ImportError:
            _print('Found no ANNarchyCython library ...')
        else:
            _print('Inited profiler.')
            self._profile_instance = ANNarchyCython.pyProfile()
            self._network = ANNarchyCython.pyNetwork()
        
        self._mean_sum = []
        self._std_dev_sum = []
        self._mean_step = []
        self._std_dev_step = []
        
        self._net_win = pg.GraphicsWindow(title="Speedup: network overall")
        self._sum_plot = self._net_win.addPlot(title = "weighted sum", axisItems = {'bottom': IntAxis('bottom') })
        self._sum_plot.setLabel('left', "computation time", units='s')
        self._sum_plot.setLabel('bottom', "number of trials",)

        self._step_plot = self._net_win.addPlot(title = "neuron step", axisItems = {'bottom': IntAxis('bottom') })
        self._step_plot.setLabel('left', "computation time", units='s')
        self._step_plot.setLabel('bottom', "number of trials",)
                        
    def measure(self, name, overwrite=True):
        """
        """
        pop = get_population(name)
        self._mean_sum.append( self._profile_instance.avgTimeSum(name, 0, 0, True) )
        self._std_dev_sum.append( self._profile_instance.stdDevSum(name, 0, 0, True) )

        self._mean_step.append( self._profile_instance.avgTimeStep(name, 0, 0, True) )
        self._std_dev_step.append( self._profile_instance.stdDevStep(name, 0, 0, True) )
        
        if overwrite:
            self._profile_instance.resetTimer()
     
    def visualize(self, error_bar = False):   
        def create_error_bar(idx, mean, std):
            """
            for equal configuration on all plots
            """
            err = pg.ErrorBarItem( x=idx, 
                                   y=mean,
                                   top=std, 
                                   bottom=std, 
                                   beam=0.5)
            return err

        x_scale = [ x for x in xrange(len(self._mean_sum))]
        self._sum_plot.plot( x_scale, self._mean_sum )
        if error_bar:
            self._sum_plot.addItem(create_error_bar(numpy.array(x_scale), numpy.array(self._mean_sum), numpy.array(self._std_dev_sum)))

        self._step_plot.plot( x_scale, self._mean_step )
        if error_bar:
            self._sum_plot.addItem(create_error_bar(numpy.array(x_scale), numpy.array(self._mean_step), numpy.array(self._std_dev_step)))
        