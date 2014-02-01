from PyQt4.QtGui import QWidget, QLineEdit, QComboBox
from PyQt4.QtCore import pyqtSlot, pyqtSignal

from GLPlotWidget import *

class VisControlWidget(QWidget):
    signal_change_grid = pyqtSignal(int, int)
    change_color_plot1d = pyqtSignal(int, int, int)
    
    def __init__(self, parent=None):
        super(VisControlWidget, self).__init__(parent)
       
    @pyqtSlot()
    def initialize(self):
        self._x_dim = self.findChild(QLineEdit, 'grid_x_dim')
        self._y_dim = self.findChild(QLineEdit, 'grid_y_dim')
        
        self.change_grid() # set grid with default values
        
        #
        # initialize plot config
        self._plot_select = self.findChild(QComboBox, 'plot_select')
        self._plot_select.addItem("None")
        self._plot_select.addItem("Curve")
        self._plot_select.addItem("2d plane")

        #
        # curve colors
        self._plot_1d_color = self.findChild(QComboBox, 'plot_1d_color')
        self._plot_1d_color.addItem("red")
        self._plot_1d_color.addItem("green")
        self._plot_1d_color.addItem("blue")

    @pyqtSlot(int)
    def change_color(self, color_id):
        
        self.change_color_plot1d.emit(0,0,id)
        
    @pyqtSlot()
    def change_grid(self):
        """
        Connected to press on ``apply`` button.
        
        Check the user input and emit it to visualizer widget
        """
        try:
            x = int(str(self._x_dim.text()))
            y = int(str(self._y_dim.text()))
        except ValueError:
            x = 1
            y = 1
             
        self.signal_change_grid.emit(x,y)
        
class VisualizerWidget(QWidget):
    """
    Embedded widget in the visualizer tab to maintain the
    several OpenGL widgets
    """
    def __init__(self, parent=None):
        super(VisualizerWidget, self).__init__(parent)
        self._gl_instances = {}
       
    @pyqtSlot()
    def initialize(self):
        pass
    
    def set_data(self, x, y, x_data, y_data):
        """
        
        """
        self._gl_instances[(y,x)].set_data(x_data, y_data)
        
    def render_data(self):
        """
        
        """
        for plot in self._gl_instances.itervalues():
            plot.render()
            
    @pyqtSlot(int, int)
    def change_grid(self, x_size, y_size):
        #
        # delete old ones
        while self.layout().count():
            item = self.layout().takeAt(0)
            widget = item.widget()
            widget.deleteLater()

        self._gl_instances = {}
 
        #
        # create new widgets
        if x_size > 1 or y_size > 1:
            for y in xrange(y_size):
                for x in xrange(x_size):
                    tmp = GLPlot1d(self)
                    self._gl_instances.update( { (y, x) : tmp})
                    self.layout().addWidget(tmp, y, x)
                    
        else:
            tmp = GLPlot1d(self)
            self._gl_instances.update( { (0,0) : tmp})
            self.layout().addWidget( tmp, 0, 0)

    