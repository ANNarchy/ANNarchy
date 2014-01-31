from PyQt4.QtGui import QStackedWidget, QLineEdit
from PyQt4.QtCore import pyqtSlot

from ANNarchy4 import *

class StackWidget(QStackedWidget):
    
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
        #
        # create short cut links to appended widgets
        self._pop_name = self.findChild(QLineEdit, 'pop_name')
        self._pop_size = self.findChild(QLineEdit, 'pop_size')
         
    @pyqtSlot(int, int)
    def update_population(self, tab, pop_id):
        self.setCurrentIndex(tab)

        try:
            pop_obj = get_population('Population'+str(pop_id))
            
            self._pop_name.setText( pop_obj.name )
            self._pop_size.setText( str(pop_obj.geometry) )
        except:
            pass
        
