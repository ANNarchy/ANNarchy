
# Visualizer
try:
    from pyqtgraph.Qt import QtGui, QtCore
    import pyqtgraph as pg
except:
    print('PyQtGraph is not installed, can not visualize the network.')
    exit(0)
    
class Viewer(object):
    " Class to visualize the network activity using PyQtGraph."
    
    def __init__(self, func):

        self.func = func
    
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
        
        self.win.show()
        

    def update(self):
        # Perform a trial
        r_i, r_f, w = self.func()
        # Refresh the GUI
        self.input_vis.setImage(r_i)
        self.feature_vis.setImage(r_f)
        self.rv_vis.setImage(w)
        # Listen to mouse/keyboard events
        QtGui.QApplication.processEvents()
        
    def run(self):
        
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(0)  
        QtGui.QApplication.instance().exec_()  