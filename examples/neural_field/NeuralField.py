#
#
#    ANNarchy-4 NeuralField
#
#
from ANNarchy4 import *

setup(dt=1.0)
#
# Define the neuron classes
#
Input = RateNeuron(   
parameters="""
    tau = 1.0
    baseline = 0.0
""",
equations="""
    noise = 0.0 #Uniform(-0.5, 0.5)
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

nb_neurons = 20

InputPop = Population((nb_neurons, nb_neurons), Input)
FocusPop = Population((nb_neurons, nb_neurons), Focus)

Proj1 = Projection( 
    pre = InputPop, 
    post = FocusPop, 
    target = 'exc'
).connect_one_to_one( weights=0.5 )

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

    # Viz
    from pyqtgraph.Qt import QtGui, QtCore
    import pyqtgraph as pg
    import pyqtgraph.widgets.RemoteGraphicsView
    
    app = pg.mkQApp()
    win = QtGui.QMainWindow()
    win.resize(800,800)
    imv = pg.ImageView()
    win.setCentralWidget(imv)
    win.show()

    
    freq = 50.0
    period = 5000
    w = InputPop.geometry[0]
    h = InputPop.geometry[1]
    
    angle = 0.0
    radius = 0.5
    sigma = 2.0
    
    data = np.zeros(w*h)
    
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)
    
    xx, yy = np.meshgrid(x, y)
    
    def update():
        
        global angle, data, InputPop, imv
        
        #angle
        angle += freq/float(period)
        
        cw = w / 2.0 * ( 1.0 + radius * np.cos(2 * np.pi * angle ) )
        ch = h / 2.0 * ( 1.0 + radius * np.sin(2 * np.pi * angle ) )
        
        data =  0.5 * np.exp(-((xx-cw)**2 + (yy-ch)**2)/2.0/sigma**2)   

        InputPop.baseline = data    
        simulate(freq)        
        
        
        imv.setImage(FocusPop.rate)
     
    update() 
                
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(0)
    QtGui.QApplication.instance().exec_()

