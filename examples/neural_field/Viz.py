# Visualizer using PyQtGraph
try:
    from pyqtgraph.Qt import QtGui, QtCore
    import pyqtgraph as pg
except:
    print('PyQtGraph is not installed on your system, can not visualize the network.')
    exit(0)
try:
    import pyqtgraph.opengl as gl
except:
    print('OpenGL is not installed on your system, can not visualize the network.')
    exit(0)

import numpy as np

class GLViewer(object):
    " Class to visualize the network activity using PyQtGraph and openGL."
    def __init__(self, populations, func, update_rate): 
        # Parameters   
        self.populations = populations
        self.func = func    
        self.update_rate = update_rate
        # Window
        self.win = gl.GLViewWidget()
        self.win.show()
        self.win.setCameraPosition(distance=40)
        # Prepare the plots
        self.plots = []
        shift = 0
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
            shift -= 25

    def scale(self, data):
        " Colors are shown in the range [-1, 1] per default."
        return 1.8 * data -0.9

    def update(self):
        "Callback"
        # Simulate for 200ms
        self.func(self.update_rate)     
        # Refresh the GUI
        for i in range(len(self.populations)):
            self.plots[i].setData(z=self.scale(self.populations[i].r)) 
        # Listen to mouse/keyboard events
        QtGui.QApplication.processEvents()

    def run(self):
        "Inifinite loop"
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(0)  
        QtGui.QApplication.instance().exec_() 


def loop_bubbles(populations, func, update_rate):
    "Launches the GL GUI and rotates the bubble infinitely."
    # Create the GUI using PyQtGraph
    app = QtGui.QApplication([])
    viewer = GLViewer(populations, func, update_rate)
    # Start the simulation forever          
    viewer.run()