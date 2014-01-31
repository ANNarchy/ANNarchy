from PyQt4.QtGui import QWidget, QLineEdit
from PyQt4.QtCore import pyqtSlot, pyqtSignal

class VisControlWidget(QWidget):
    signal_change_grid = pyqtSignal(int, int)
    
    def __init__(self, parent=None):
        super(VisControlWidget, self).__init__(parent)
       
    @pyqtSlot()
    def initialize(self):
        self._x_dim = self.findChild(QLineEdit, 'grid_x_dim')
        self._y_dim = self.findChild(QLineEdit, 'grid_y_dim')

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
    def __init__(self, parent=None):
        super(VisualizerWidget, self).__init__(parent)
        print 'Test'
       
    @pyqtSlot()
    def initialize(self):
        print self.layout() 
        pass
    
    @pyqtSlot(int, int)
    def change_grid(self, x_size, y_size):
        print x_size, y_size
    