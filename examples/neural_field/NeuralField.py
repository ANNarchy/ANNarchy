#   Simple example of a neural field
#
#   authors: Julien Vitay, Helge Uelo Dinkelbach
#
from ANNarchy import *

setup(dt=1.0)

# Define the neuron classes
InputNeuron = RateNeuron(   
    parameters="""
        baseline = 0.0
    """,
    equations="""
        noise = Uniform(-0.5, 0.5)
        rate = pos(baseline + noise)
    """ 
)

NeuralFieldNeuron = RateNeuron(
    parameters=""" 
        tau = 10.0 : population
    """,
    equations="""
        noise = Uniform(-0.5, 0.5)
        tau * dmp / dt + mp = sum(exc) + sum(inh) + noise
        rate = if mp < 1.0 : pos(mp) else: 1.0 
    """
)

# Create the populations
nb_neurons = 20
InputPop = Population(name = 'Input', geometry = (nb_neurons, nb_neurons), neuron = InputNeuron)
FocusPop = Population(name = 'Focus', geometry = (nb_neurons, nb_neurons), neuron = NeuralFieldNeuron)

# Create the projections
input_focus = Projection( 
    pre = InputPop, 
    post = FocusPop, 
    target = 'exc'
).connect_one_to_one( weights=1.0 )

focus_focus = Projection(
    pre = FocusPop, 
    post = FocusPop, 
    target = 'inh'     
).connect_dog(    
    amp_pos=0.2, 
    sigma_pos=0.1, 
    amp_neg=0.1, 
    sigma_neg=0.7                    
)

# Visualizer using PyQtGraph
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg 
import pyqtgraph.opengl as gl

class GLViewer(object):
    " Class to visualize the network activity using PyQtGraph and openGL."
    def __init__(self, populations, world):    
        self.populations = populations
        self.world = world          
        self.win = gl.GLViewWidget()
        self.win.show()
        self.win.setCameraPosition(distance=50)
        self.plots = []
        shift = - 20
        for pop in self.populations: 
            p = gl.GLSurfacePlotItem(
                x = np.linspace(0, pop.geometry[0]-1, pop.geometry[0]), 
                y = np.linspace(0, pop.geometry[1]-1, pop.geometry[1]), 
                shader='heightColor', 
                computeNormals=False, 
                smooth=False
            )
            p.translate(shift, -10, -1)
            self.win.addItem(p)
            self.plots.append(p)
            shift += 25
    def scale(self, data):
        " Colors are shown in the range [-1, 1] per default."
        return 1.8 * data -0.9
    def update(self):
        # Simulate for 200ms
        self.world.rotate(200)      
        # Actualize the GUI
        for i in range(len(self.populations)):
            self.plots[i].setData(z=self.scale(self.populations[i].rate)) 
        # Listen to mouse/keyboard events
        QtGui.QApplication.processEvents()
    def run(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(0)  
        QtGui.QApplication.instance().exec_() 
        
 

# Main program
if __name__ == "__main__":

    # Analyse and compile everything, initialize the parameters/variables...
    compile()   
    
    # Import the environment for the simulation (Cython)
    import pyximport; pyximport.install()
    from BubbleWorld import World
    world = World(pop = InputPop, radius = 0.5, sigma = 2.0, period = 5000.0)

    # Create the GUI using PyQtGraph
    app = QtGui.QApplication([])
    viewer = GLViewer(populations = [InputPop, FocusPop], world=world)
    # Start the simulation            
    viewer.run()

     


