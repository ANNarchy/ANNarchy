#   Bar Learning example
#
#   authors: Julien Vitay, Helge Uelo Dinkelbach

from ANNarchy import *

setup()

# Defining the neuron
InputNeuron = RateNeuron(   
    parameters="""
        rate = 0.0
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
        min_value = 0.0 : postsynaptic
    """,
    equations="""
        tau * dvalue/dt = pre.rate * post.rate - alpha * post.rate^2 * value : min=min_value
    """
)  


# Creating the populations
Input = Population(geometry=(8,10), neuron=InputNeuron)
Feature = Population(geometry=(8, 4), neuron=LeakyNeuron)

# Creating the projections
Input_Feature = Projection(
    pre=Input, 
    post=Feature, 
    target='exc', 
    synapse = Oja    
).connect_all_to_all( weights = Uniform(-0.5, 0.5) )
Input_Feature.min_value = -10.0
                     
Feature_Feature = Projection(
    pre=Feature, 
    post=Feature, 
    target='inh', 
    synapse = Oja
).connect_all_to_all( weights = Uniform(0.0, 1.0) )
Feature_Feature.alpha = 0.3


# Definition of the environment
def set_input():
    # Reset the firing rate for all neurons
    Input.rate = 0.0
    # Clamp horizontal bars
    for h in range(Input.geometry[0]):
        if np.random.random() < 1.0/ float(Input.geometry[0]):
            Input[h, :].rate = 1.0
    # Clamp vertical bars
    for w in range(Input.geometry[1]):
        if np.random.random() < 1.0/ float(Input.geometry[1]):
            Input[:, w].rate = 1.0
    

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
        self.input_vis.setImage(Input.rate.T)
        self.feature_vis.setImage(Feature.rate)
        self.rv_vis.setImage(Input_Feature._gather_data('value').transpose())
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
