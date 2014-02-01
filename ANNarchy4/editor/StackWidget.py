from PyQt4.QtGui import QStackedWidget, QLineEdit
from PyQt4.QtCore import pyqtSlot, pyqtSignal

from ANNarchy4 import *

class StackWidget(QStackedWidget):
    signal_update_population = pyqtSignal(int)
    
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
             
    @pyqtSlot(int, int)
    def update_population(self, tab, pop_id):
        self.setCurrentIndex(tab)
        
        if tab == 1:
            self.signal_update_population.emit( pop_id )