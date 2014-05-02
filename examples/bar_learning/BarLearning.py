#   Bar Learning example
#
#   authors: Julien Vitay, Helge Uelo Dinkelbach

from ANNarchy import *

# Defining the neuron
InputNeuron = RateNeuron(   
    parameters="""
        baseline = 0.0
    """,
    equations="""
        rate = baseline
    """ 
)

LeakyNeuron = RateNeuron(
    parameters=""" 
        tau = 10.0 : population
    """,
    equations="""
        tau * drate/dt + rate = sum(exc) - sum(inh) : min=0.0
    """
)

# Defining the synapse
Oja = RateSynapse(
    parameters=""" 
        tau = 2000.0 : postsynaptic
        alpha = 8.0 : postsynaptic
    """,
    equations="""
        tau * dvalue/dt = pre.rate * post.rate - alpha * post.rate^2 * value : min=-100.0
    """
)  

AntiHebb = RateSynapse(
    parameters=""" 
        tau = 2000.0 : postsynaptic
        alpha = 0.3 : postsynaptic
    """,
    equations="""
        tau * dvalue/dt = pre.rate * post.rate - alpha * post.rate^2 * value : min = 0.0
    """
)  

# Creating the populations
input_pop = Population(geometry=(8,8), neuron=InputNeuron)
feature_pop = Population(geometry=(8, 4), neuron=LeakyNeuron)

# Creating the projections
input_feature = Projection(
    pre=input_pop, 
    post=feature_pop, 
    target='exc', 
    synapse = Oja    
).connect_all_to_all( weights = Uniform(-0.5, 0.5) )
                     
feature_feature = Projection(
    pre=feature_pop, 
    post=feature_pop, 
    target='inh', 
    synapse = AntiHebb
).connect_all_to_all( weights = Uniform(0.0, 1.0) )
feature_feature.alpha = 0.3


# Definition of the environment
def set_input():
    # Choose which bars will be used as inputs
    values = np.zeros(input_pop.geometry)
    for w in range(input_pop.geometry[1]):
        if np.random.random() < 1./ float(input_pop.geometry[1]):
            values[:, w] = 1.
    for h in range(input_pop.geometry[0]):
        if np.random.random() < 1./ float(input_pop.geometry[0]):
            values[h, :] = 1.
    # Set the input
    input_pop.baseline = values
    

# Visualizer
try:
    from pyqtgraph.Qt import QtGui, QtCore
    import pyqtgraph as pg
except:
    print 'PyQtGraph is not installed, can not visualize the network.'
    exit(0)
    
class Viewer(object):
    " Class to visualize the network activity using PyQtGraph."
    
    def __init__(self):
    
        app = pg.mkQApp()
        self.win = pg.GraphicsWindow(title="Bar learning")
        self.win.resize(800,800)
        self.win.addLabel("Input")
        self.win.addLabel("Feature")
        self.win.nextRow()
        
        box = self.win.addViewBox(lockAspect=True)
        self.input_vis = pg.ImageItem()
        box.addItem(self.input_vis)
        
        box = self.win.addViewBox(lockAspect=True)
        self.feature_vis = pg.ImageItem()
        box.addItem(self.feature_vis)
        self.win.nextRow()
        
        self.win.addLabel("Receptive fields", colspan=2)
        self.win.nextRow()
        box = self.win.addViewBox(lockAspect=True, colspan=2)
        self.rv_vis = pg.ImageItem()
        box.addItem(self.rv_vis)
        
        self.win.show()
        

    def update(self):
        # Simulate for 50 ms with a new input
        set_input()
        simulate(50) 
        # Refresh the GUI
        self.input_vis.setImage(input_pop.rate)
        self.feature_vis.setImage(feature_pop.rate)
        self.rv_vis.setImage(input_feature._gather_data('value').transpose())
        # Listen to mouse/keyboard events
        QtGui.QApplication.processEvents()
        
    def run(self):
        
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(0)  
        QtGui.QApplication.instance().exec_()     
        
if __name__=='__main__':

    compile()

    # Create the GUI
    view = Viewer()
    
    # Start the simulation
    view.run()

        
