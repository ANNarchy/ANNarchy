from PyQt4.QtGui import QTabWidget, QMessageBox
from PyQt4.QtCore import pyqtSlot, QString, QObject, SIGNAL, SLOT

class MyTabWidget(QTabWidget):
    
    def __init__(self, parent=None):
        super(MyTabWidget, self).__init__(parent)

    @pyqtSlot()
    def initialize(self):  
        pass
                