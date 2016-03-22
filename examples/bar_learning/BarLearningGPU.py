#   Bar Learning example
#
#   authors: Julien Vitay, Helge Uelo Dinkelbach

from ANNarchy import *

setup(paradigm="cuda")

# Defining the neuron
InputNeuron = Neuron(   
    parameters="""
        r = 0.0
    """
)

LeakyNeuron = Neuron(
    parameters=""" 
        tau = 10.0 : population
    """,
    equations="""
        tau * dr/dt + r = sum(exc) - sum(inh) : min=0.0
    """
)

# Defining the synapse
Oja = Synapse(
    parameters=""" 
        tau = 2000.0 : postsynaptic
        alpha = 8.0 : postsynaptic
        min_w = 0.0 : postsynaptic
    """,
    equations="""
        tau * dw/dt = pre.r * post.r - alpha * post.r^2 * w : min=min_w
    """
)  


# Creating the populations
Input = Population(geometry=(8, 8), neuron=InputNeuron)
#Input = Population(geometry=(64, 64), neuron=InputNeuron)
Feature = Population(geometry=(8, 4), neuron=LeakyNeuron)
#Feature = Population(geometry=(32, 8), neuron=LeakyNeuron)

# Creating the projections
Input_Feature = Projection(
    pre=Input, 
    post=Feature, 
    target='exc', 
    synapse = Oja    
).connect_all_to_all( weights = Uniform(-0.5, 0.5) )
Input_Feature.min_w = -10.0
                     
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
    Input.r = 0.0
    # Clamp horizontal bars
    for h in range(Input.geometry[0]):
        if np.random.random() < 1.0/ float(Input.geometry[0]):
            Input[h, :].r = 1.0
    # Clamp vertical bars
    for w in range(Input.geometry[1]):
        if np.random.random() < 1.0/ float(Input.geometry[1]):
            Input[:, w].r = 1.0
    

# Visualizer
try:
    from pyqtgraph.Qt import QtGui, QtCore
    import pyqtgraph as pg
except:
    print('PyQtGraph is not installed, can not visualize the network.')
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
        
        self.count = 0
        self.update_period = 200 # every 200 input presentations update weight plot
        self.win.show()
        

    def update(self):
        # Simulate for 50 ms with a new input
        set_input()
        simulate(50)
        self.count += 1

        # Refresh the GUI
        self.input_vis.setImage(Input.r)
        self.feature_vis.setImage(Feature.r)
        
        if self.count == self.update_period:
            self.rv_vis.setImage(Input_Feature.receptive_fields())
            self.count = 0

        # Listen to mouse/keyboard events
        QtGui.QApplication.processEvents()
        
    def run(self):
        
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(0)  
        QtGui.QApplication.instance().exec_()     
        
if __name__=='__main__':

    config = {
        'device': 0,
        Input: { 'num_threads': 32, 'stream': 0 },
        Feature: { 'num_threads': 32, 'stream': 1 },
        Input_Feature: { 'num_threads': 192, 'stream': 0 },
        Feature_Feature: { 'num_threads': 96, 'stream': 1 }
    }

    compile(cuda_config=config)

    # Create the GUI
    view = Viewer()
    
    # Start the simulation
    view.run()
