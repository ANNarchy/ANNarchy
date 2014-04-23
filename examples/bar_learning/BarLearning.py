#
#   BarLearning example for ANNarchy4
#
#   authors: Julien Vitay, Helge Uelo Dinkelbach
#
from ANNarchy4 import *

# Defining the neurons
InputNeuron = RateNeuron(
    parameters=""" 
        tau = 10.0 : population
        baseline = 0.0 
    """,
    equations="""
        tau * drate/dt + rate = baseline : min=0.0
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

# Defining the synapses
Oja = RateSynapse(
    parameters=""" 
        tau = 2000.0 : postsynaptic
        alpha = 8.0 : postsynaptic
    """,
    equations="""
        tau * dvalue/dt = pre.rate * post.rate - alpha * post.rate^2 * value
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
nb_neurons = 16
input_pop = Population(geometry=(nb_neurons, nb_neurons), neuron=InputNeuron)

# the division of two is just for a more quadratic visualization of receptive fields
feature_pop = Population(geometry=(nb_neurons/2, 4*2), neuron=LeakyNeuron)

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

# Definition of the environment
def set_input():
    # Choose which bars will be used as inputs
    values = np.zeros((nb_neurons, nb_neurons))
    for w in range(nb_neurons):
        if np.random.random() < 1./ float(nb_neurons):
            values[:, w] = 1.
    for h in range(nb_neurons):
        if np.random.random() < 1./ float(nb_neurons):
            values[h, :] = 1.

    # Set the input
    input_pop.baseline = values.reshape(nb_neurons**2)

if __name__=='__main__':

    #compile(cpp_stand_alone=True)
    compile()
    #input_feature.save_connectivity_as_csv()
    #feature_feature.save_connectivity_as_csv()
    #exit(0)

    from pyqtgraph.Qt import QtGui, QtCore
    import pyqtgraph as pg
    
    app = pg.mkQApp()

    win = pg.GraphicsWindow(title="Bar learning ("+str(nb_neurons)+" neurons)")
    win.resize(800,800)

    win.addLabel("Input")
    win.addLabel("Feature")

    win.nextRow()

    box = win.addViewBox(lockAspect=True)
    input_vis = pg.ImageItem()
    box.addItem(input_vis)

    box = win.addViewBox(lockAspect=True)
    feature_vis = pg.ImageItem()
    box.addItem(feature_vis)

    win.nextRow()
    win.addLabel("Receptive field")

    win.nextRow()
    box = win.addViewBox(lockAspect=True)
    rv_vis = pg.ImageItem()
    box.addItem(rv_vis)
    
    #win.setCentralWidget(imv)
    win.show()

    def update():
        for trial in range(3000):
            if (trial > 0) and (trial % 100==0):
                print trial
                
            set_input()
            simulate(50) 
     
            input_vis.setImage(input_pop.baseline)
            feature_vis.setImage(feature_pop.rate)
            rv_vis.setImage(input_feature._gather_data('value').transpose())
            QtGui.QApplication.processEvents()
                
        timer.stop()
        
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(0)
    QtGui.QApplication.instance().exec_()
