from PyQt4.QtGui import QStackedWidget, QLineEdit
from PyQt4.QtCore import pyqtSlot, pyqtSignal, QString

from ANNarchy4 import *

class StackWidget(QStackedWidget):
    signal_update_population = pyqtSignal(QString, int)
    signal_update_projection = pyqtSignal(QString, int)
    
    def __init__(self, parent=None):
        """
        Constructor called from Ui_ANNarchyEditor class.
         
        .. hint: At this point no widget related initialization is possible. Use the ``initialize`` function for these things.
        """
        super(StackWidget, self).__init__(parent)
       
    @pyqtSlot()
    def initialize(self):
        """
        Initialization of links etc, after full initialization of ANNarchyEditor.
        """
        pass
             
    @pyqtSlot(QString, int, int)
    def update_population(self, net_name, tab, pop_id):
        """
        Update either population or projection view depending on selected objects. 
        """
        self.setCurrentIndex(tab)
        
        if tab == 1:
            self.signal_update_population.emit( net_name, pop_id )
        elif tab == 2:
            self.signal_update_projection.emit( net_name, pop_id )