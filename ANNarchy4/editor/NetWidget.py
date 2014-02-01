from PyQt4.QtGui import QListView, QInputDialog, QStandardItemModel, QStandardItem, QMessageBox, QWidget
from PyQt4.QtCore import SIGNAL, SLOT, pyqtSlot, pyqtSignal, QString, QObject

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
        self._model = QStandardItemModel(self)
        self._model.appendRow(QStandardItem("<Press here to add ...>"))

        for name in self._rep.get_entries('network'):
            self._model.appendRow(QStandardItem( name ))            
        self.setModel(self._model)

        self.connect(self,SIGNAL("clicked(QModelIndex)"), self, SLOT("ItemClicked(QModelIndex)"))

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
            self._rep.add_object('network', text, {})
            
            idx = self._model.index(self._model.rowCount()-1, 0)
            self.setCurrentIndex( idx )
            
class PopView(QWidget):
    
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)

    def set_repository(self, repo=None):
        self._rep = repo
        
    @pyqtSlot()
    def initialize(self):
        pass
    