from PyQt4.QtGui import QListView, QInputDialog, QStandardItemModel, QStandardItem, QMessageBox
from PyQt4.QtCore import SIGNAL, SLOT, pyqtSlot, pyqtSignal, QString

class ListView(QListView):
    _update = pyqtSignal(str)
    
    def __init__(self, parent):
        super(QListView,self).__init__(parent)
        
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
    signal_show_template = pyqtSignal(int, QString)
    
    def __init__(self, parent):
        super(ListView, self).__init__(parent)
        
    @pyqtSlot()
    def initialize(self):

        self._model = QStandardItemModel(self)
        self._model.appendRow(QStandardItem("<Press here to add ...>"))
        self.setModel(self._model)
        
        self.connect(self,SIGNAL("clicked(QModelIndex)"), self, SLOT("ItemClicked(QModelIndex)"))

    @pyqtSlot("QString")
    def add_entry(self, name):
        self._model.appendRow(QStandardItem(name))
        
    @pyqtSlot("QModelIndex")
    def ItemClicked(self, index):
        if index.data().toString() == "<Press here to add ...>":
            self.input_dialog()
            
        self.signal_show_template.emit( 0, self.currentIndex().data().toString() )

    def input_dialog(self):
        text, ok = QInputDialog.getText(self, 'New neuron definition', 'Enter neuron type name:')
        
        if ok:
            if self._model.findItems(text) != []:
                QMessageBox.warning(self,"Add "+_type, "The "+self._type+" is already existing")
                return
            
            item = QStandardItem( text )
            self._model.appendRow(item)
            self.setModel(self._model)
            
            idx = self._model.index(self._model.rowCount()-1, 0)
            self.setCurrentIndex( idx )
