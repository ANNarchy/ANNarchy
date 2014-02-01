from PyQt4.QtGui import QListView, QInputDialog, QStandardItemModel, QStandardItem, QMessageBox
from PyQt4.QtCore import SIGNAL, SLOT, pyqtSlot, pyqtSignal, QString, QObject


class NeuronListView(QListView):
    signal_show_template = pyqtSignal(int, QString)
    
    def __init__(self, parent):
        super(QListView, self).__init__(parent)

    def set_repository(self, repo=None):
        self._rep = repo
        
    @pyqtSlot()
    def initialize(self):

        self._model = QStandardItemModel(self)
        self._model.appendRow(QStandardItem("<Press here to add ...>"))
        for name in self._rep.get_entries('neuron'):
            self._model.appendRow(QStandardItem( name ))            
        
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
                QMessageBox.warning(self,"Add neuron", "The neuron type is already existing")
                return
            
            item = QStandardItem( text )
            self._model.appendRow(item)
            self.setModel(self._model)
            
            idx = self._model.index(self._model.rowCount()-1, 0)
            self.setCurrentIndex( idx )

class SynapseListView(QListView):
    signal_show_template = pyqtSignal(int, QString)
    
    def __init__(self, parent):
        super(QListView, self).__init__(parent)

    def set_repository(self, repo=None):
        self._rep = repo
        
    @pyqtSlot()
    def initialize(self):
        self._model = QStandardItemModel(self)
        self._model.appendRow(QStandardItem("<Press here to add ...>"))

        for name in self._rep.get_entries('synapse'):
            self._model.appendRow(QStandardItem( name ))            
        self.setModel(self._model)
        
        self.connect(self,SIGNAL("clicked(QModelIndex)"), self, SLOT("ItemClicked(QModelIndex)"))

    @pyqtSlot("QString")
    def add_entry(self, name):
        self._model.appendRow(QStandardItem(name))
        
    @pyqtSlot("QModelIndex")
    def ItemClicked(self, index):
        if index.data().toString() == "<Press here to add ...>":
            self.input_dialog()
            
        self.signal_show_template.emit( 2, self.currentIndex().data().toString() )

    def input_dialog(self):
        text, ok = QInputDialog.getText(self, 'New synapse definition', 'Enter synapse type name:')
        
        if ok:
            if self._model.findItems(text) != []:
                QMessageBox.warning(self,"Add synapse", "The synapse type is already existing")
                return
            
            item = QStandardItem( text )
            self._model.appendRow(item)
            self.setModel(self._model)
            
            idx = self._model.index(self._model.rowCount()-1, 0)
            self.setCurrentIndex( idx )

class NetworkListView(QListView):
    signal_show_network = pyqtSignal(QString)
    
    def __init__(self, parent):
        super(QListView, self).__init__(parent)
        self._model = QStandardItemModel(self)
        self._model.appendRow(QStandardItem("<Press here to add ...>"))
        self.setModel(self._model)

    def set_repository(self, repo=None):
        self._rep = repo
        
    @pyqtSlot()
    def initialize(self):
        self.connect(self,SIGNAL("clicked(QModelIndex)"), self, SLOT("ItemClicked(QModelIndex)"))

    @pyqtSlot("QString")
    def add_entry(self, name):
        self._model.appendRow(QStandardItem(name))
        self._rep.add_object('network', name)
        
    @pyqtSlot("QModelIndex")
    def ItemClicked(self, index):
        if index.data().toString() == "<Press here to add ...>":
            self.input_dialog()
            
        self.signal_show_network.emit( self.currentIndex().data().toString() )

    def input_dialog(self):
        text, ok = QInputDialog.getText(self, 'New network definition', 'Enter network name:')
        
        if ok:
            if self._model.findItems(text) != []:
                QMessageBox.warning(self,"Add network", "The network is already existing")
                return
            
            item = QStandardItem( text )
            self._model.appendRow(item)
            self.setModel(self._model)
            
            idx = self._model.index(self._model.rowCount()-1, 0)
            self.setCurrentIndex( idx )
