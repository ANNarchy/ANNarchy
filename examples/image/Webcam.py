from ANNarchy import *
from ANNarchy.extensions.image import *
try:
    from pyqtgraph.Qt import QtGui, QtCore
    import pyqtgraph as pg
except:
    print 'PyQtGraph is not installed, can not visualize the network.'
    exit(0)

# Visualizer    
class Viewer(object):
    " Class to visualize the network activity using PyQtGraph."
    
    def __init__(self, pop):
        self.pop = pop
        app = pg.mkQApp()
        self.win = pg.GraphicsWindow(title="Live webcam")
        self.win.resize(640,480)        
        box = self.win.addViewBox(lockAspect=True)
        box.invertY()
        self.vis = pg.ImageItem()
        box.addItem(self.vis)        
        self.win.show()
        
        self.lastUpdate = pg.ptime.time()
        self.avgFps = 0.0
        

    def update(self):
        # Simulate for 10 ms with a new input
        self.pop.grab_image()
        # Refresh the GUI
        self.vis.setImage(np.swapaxes(self.pop.r,0,1))
        # Listen to mouse/keyboard events
        QtGui.QApplication.processEvents()
        # FPS
        now = pg.ptime.time()
        fps = 1.0 / (now - self.lastUpdate)
        self.lastUpdate = now
        self.avgFps = self.avgFps * 0.8 + fps * 0.2
        print self.avgFps
        
    def run(self):
        
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(0)  
        QtGui.QApplication.instance().exec_() 
        timer.stop()

if __name__ == '__main__':

    # Create the population    
    #pop = VideoPopulation(name='test', geometry=(480, 640, 3))
    #pop = VideoPopulation(name='test', geometry=(480, 640, 1))
    pop = VideoPopulation(name='test', geometry=(240, 320, 3))
    
    # Compile
    compile()
    
    # Start the camera        
    pop.start_camera(0)
    
    # Start the GUI
    view = Viewer(pop)
    view.run()    
