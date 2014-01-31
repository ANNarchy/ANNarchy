from PyQt4.QtGui import QTabWidget, QMessageBox
from PyQt4.QtCore import pyqtSlot, QString, QObject, SIGNAL, SLOT

class MyTabWidget(QTabWidget):
    
    def __init__(self, parent=None):
        super(MyTabWidget, self).__init__(parent)
        
        QObject.connect(self, SIGNAL("currentChanged(int)"), self, SLOT("tabChangedSlot(int)"))
        
    @pyqtSlot(int)
    def tabChangedSlot(self,argTabIndex):
        print argTabIndex
        #QMessageBox.information(self, "Tab Index Changed!", "Current Tab Index: "+QString.number(argTabIndex));
        #reply = QMessageBox.question(self, 'Message', "There are unsaved changes, you want to save?", QMessageBox.Yes, QMessageBox.No)
    
        #if reply == QMessageBox.No:
        #    print 'xxx'