#
#
#    ANNarchy-4 NeuralField
#
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
    noise = Uniform(-0.1, 0.1)
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

# Main program
if __name__ == "__main__":

    # Analyse and compile everything, initialize the parameters/variables...
    compile()   

    # Create the GUI using PyQtGraph
    from pyqtgraph.Qt import QtGui, QtCore
    import pyqtgraph as pg    
    app = pg.mkQApp()
    win = QtGui.QMainWindow()
    win.resize(800,800)
    imv = pg.ImageView()
    win.setCentralWidget(imv)
    win.show()

    
    # Import the envorinment for the simulation (Cython)
    import pyximport; pyximport.install()
    from BubbleWorld import World
    world = World(pop = InputPop, radius = 0.5, sigma = 2.0, period = 5000.0)
    
    # Method called regularly by the GUI
    def update():
        # Simulate for 200ms
        world.rotate(200)      
        # Actualize the GUI
        imv.setImage(FocusPop.rate)
        QtGui.QApplication.processEvents()
     
    # Start the simulation            
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(0)
    QtGui.QApplication.instance().exec_()

