#
#
#   ANNarchy-4 NeuralField
#
#   authors: Julien Vitay, Helge Uelo Dinkelbach
#
from ANNarchy4 import *

setup(dt=1.0)

# Define the neuron classes
Input = RateNeuron(   
parameters="""
    tau = 1.0
    baseline = 0.0
""",
equations="""
    noise = Uniform(-0.5, 0.5)
    tau * dmp / dt + mp = baseline + noise
    rate = pos(mp)
""" 
)

Focus = RateNeuron(
parameters=""" 
    tau = 20.0
    noise = 0.0
    baseline = 0.0
    threshold_min = 0.0
    threshold_max = 1.0
""",
equations="""
    tau * dmp / dt + mp = sum(exc) + sum(inh) + baseline + noise
    rate = if mp > threshold_max : threshold_max else: pos(mp) : init = 0.0
"""
)

# Create the populations
nb_neurons = 20
InputPop = Population(name = 'Input', geometry = (nb_neurons, nb_neurons), neuron = Input)
FocusPop = Population(name = 'Focus', geometry = (nb_neurons, nb_neurons), neuron = Focus)

# Create the projections
Proj1 = Projection( 
    pre = InputPop, 
    post = FocusPop, 
    target = 'exc'
).connect_one_to_one( weights=1.0 )

Proj2 = Projection(
    pre = FocusPop, 
    post = FocusPop, 
    target = 'inh'     
).connect_dog(    
    amp_pos=0.2, 
    sigma_pos=0.1, 
    amp_neg=0.1, 
    sigma_neg=0.7                    
)

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg 

class Viewer(object):
    def __init__(self, pop, world):
    
        self.pop = pop
        self.world = world
          
        self.win = pg.GraphicsLayoutWidget()
        self.win.resize(800,800)
        self.win.show()
        self.view = self.win.addViewBox()
        self.view.setAspectLocked(True)
        self.img = pg.ImageItem(border='w')
        self.view.addItem(self.img)
        self.view.setRange(QtCore.QRectF(0, 0, 20, 20))
    
    def update(self):
    
        # Simulate for 200ms
        self.world.rotate(200)      
        # Actualize the GUI
        self.img.setImage(self.pop.rate)
        QtGui.QApplication.processEvents()
        
    def run(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(0)  
        QtGui.QApplication.instance().exec_() 

import pyqtgraph.opengl as gl
class GLViewer(object):
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
                #y = np.array(range(pop.geometry[1])), 
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

     


