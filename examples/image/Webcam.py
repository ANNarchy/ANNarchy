from ANNarchy import *
from ANNarchy.extensions.image import *
from ANNarchy.extensions.convolution import Convolution, Pooling


# Definition of the neurons
Linear = Neuron(equations="r=sum(exc): min=0.0")
DNF = Neuron(parameters="tau=10.0", equations="tau*dr/dt + r = sum(exc) + sum(inh): min=0.0, max=1.0")


# Population getting the video stream   
width = 640
height = 480
video = VideoPopulation(geometry=(height, width, 3))

# Define a normalizedred filter with dimensions 10*10*3
extent = 10
red_filter = [[ [2.0/extent**2, -1.0/extent**2, -1.0/extent**2] for j in range(extent) ] for i in range(extent)] 

# Create a population of DNF neurons downscaling the image with a factor 10 
dnf = Population(geometry=(height/extent, width/extent), neuron = DNF)

# Create the convolution usinf the red filter
ff = Convolution(pre=video, post=dnf, target='exc').connect_filter(weights=red_filter)

# Create difference of Gaussians lateral connections for denoising/competition
lat = Projection(pre=dnf, post=dnf, target='inh').connect_dog(amp_pos = 0.15, sigma_pos = 0.05, amp_neg = 0.1, sigma_neg = 0.5, limit=0.1)

# Compile
compile()

# Start the camera        
video.start_camera(0)

# Visualize the images using PyQtGraph
try:
    from pyqtgraph.Qt import QtGui, QtCore
    import pyqtgraph as pg
except:
    print('PyQtGraph is not installed, can not visualize the network.')
    exit(0)

# Wrapping class    
class Viewer(object):
    " Class to visualize the network activity using PyQtGraph."
    
    def __init__(self, video, result):
        self.video = video
        self.result = result
        app = pg.mkQApp()
        self.win = pg.GraphicsWindow(title="Live webcam")
        self.win.resize(640,480)   

        box = self.win.addViewBox(lockAspect=True)
        box.invertY()
        self.vis = pg.ImageItem()
        box.addItem(self.vis)  
             
        box = self.win.addViewBox(lockAspect=True)
        box.invertY()
        self.res = pg.ImageItem()
        box.addItem(self.res)  

        self.win.show()
        
        self.lastUpdate = pg.ptime.time()
        self.avgFps = 0.0
        

    def update(self):
        # Set the input
        self.video.grab_image()
        # Simulate for 10 ms with a new input
        simulate(5.0)
        # Refresh the GUI
        self.vis.setImage(np.swapaxes(self.video.r,0,1))
        self.res.setImage(np.swapaxes(self.result.r,0,1))
        # Listen to mouse/keyboard events
        QtGui.QApplication.processEvents()
        # FPS
        now = pg.ptime.time()
        fps = 1.0 / (now - self.lastUpdate)
        self.lastUpdate = now
        self.avgFps = self.avgFps * 0.8 + fps * 0.2
        print(self.avgFps)
        
    def run(self):
        
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(0)  
        QtGui.QApplication.instance().exec_() 
        timer.stop()

# Start the GUI
view = Viewer(video, dnf)
view.run()    
