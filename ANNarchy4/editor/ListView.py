from PyQt4.QtGui import QListView, QInputDialog, QStandardItemModel, QStandardItem, QMessageBox
from PyQt4.QtCore import SIGNAL, SLOT, pyqtSlot, pyqtSignal

class ListView(QListView):
    _update = pyqtSignal(str)
    
    def __init__(self, parent, main_window, type):
        super(QListView,self).__init__(parent)
        
        self._type = type

        
    def initialize(self, entries=[]):
        
        self._model = QStandardItemModel(self)
        
        # user elements
        for it in entries:
            item = QStandardItem(it)   
            item.setCheckable(True) 
            self._model.appendRow(item)
        
        self.setModel(self._model)
        
        self.connect(self,SIGNAL("clicked(QModelIndex)"), self, SLOT("ItemClicked(QModelIndex)"))
        
    @pyqtSlot("QModelIndex")
    def ItemClicked(self, index):
        self._update.emit(index.data().toString())
            
class NeuronListView(ListView):
    
    def __init__(self, parent, main_window, type):
        super(ListView,self).__init__(parent, main_window, type)
        
        
    def initialize(self, entries=[]):
        
        self._model = QStandardItemModel(self)
        
        # user elements
        for it in entries:
            item = QStandardItem(it)   
            item.setCheckable(True) 
            self._model.appendRow(item)
        
        item = QStandardItem("<Press here to add ...>")
        
        self.setModel(self._model)
        
        self.connect(self,SIGNAL("clicked(QModelIndex)"), self, SLOT("ItemClicked(QModelIndex)"))
        
    def input_dialog(self):
        text, ok = QInputDialog.getText(self, 'New '+self._type, 'Enter '+self._type+' name:')
        
        if ok:
            if self._model.findItems(text) != []:
                QMessageBox.warning(self,"Add "+_type, "The "+self._type+" is already existing")
                return
            
            item = QStandardItem( text )
            item.setCheckable(True)    
            #print 'Add new population:', text
            self._model.appendRow(item)
            self.setModel(self._model)
                    
    @pyqtSlot("QModelIndex")
    def ItemClicked(self, index):
        if index.data().toString() == "<Press here to add ...>":
            self.input_dialog()
        else:
            self._update.emit(index.data().toString())            