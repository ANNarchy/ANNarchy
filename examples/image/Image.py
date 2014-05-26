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
        

    def update(self):
        # Simulate for 50 ms with a new input
        self.pop.grab_image()
        # Refresh the GUI
        self.vis.setImage(np.swapaxes(self.pop.r,0,1))
        # Listen to mouse/keyboard events
        QtGui.QApplication.processEvents()
        
    def run(self):
        
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(0)  
        QtGui.QApplication.instance().exec_() 
        timer.stop()

if __name__ == '__main__':
    # Create the population    
    pop = ImagePopulation(name='test', geometry=(480, 640))
    
    # Compile
    compile()
    
    # Set the image
    pop.set_image('test.jpg')
    
    # Visualize with Matplotlib
    import pylab as plt
    import matplotlib.image as mpimg

    fig = plt.figure()
    if pop.dimension == 3: # color
        ax = fig.add_subplot(221)
        ax.imshow(mpimg.imread('test.jpg'))
        ax.set_title('Original')
        ax = fig.add_subplot(222)
        ax.imshow(pop.r[:,:,0], cmap='gray', vmin= 0.0, vmax=1.0)
        ax.set_title('pop.r R')
        ax = fig.add_subplot(223)
        ax.imshow(pop.r[:,:,1], cmap='gray', vmin= 0.0, vmax=1.0)
        ax.set_title('pop.r G')
        ax = fig.add_subplot(224)
        ax.imshow(pop.r[:,:,2], cmap='gray', vmin= 0.0, vmax=1.0)
        ax.set_title('pop.r B')
    else: # grayscale
        ax = fig.add_subplot(121)
        ax.imshow(mpimg.imread('test.jpg'))
        ax.set_title('Original')
        ax = fig.add_subplot(122)
        ax.imshow(pop.r, cmap='gray', vmin= 0.0, vmax=1.0)
        ax.set_title('pop.r')
    plt.show()