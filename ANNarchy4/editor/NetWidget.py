from PyQt4.QtGui import QListView, QInputDialog, QLineEdit, QStandardItemModel, QStandardItem, QMessageBox, QWidget, QComboBox
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
    """
    Visualizes all the information of a selected population.
    """
    def __init__(self, parent):
        """
        Constructor.
        """
        super(QWidget, self).__init__(parent)

        self._population_data = {}

    def set_repository(self, repo):
        """
        Set the link to the application wide accessible object repository. 
        """
        self._rep = repo
        self._pop_id = -1
        
    @pyqtSlot()
    def initialize(self):
        """
        Initialization of links etc, after full initialization of ANNarchyEditor.
        
        Emitted by:
        
        *ANNarchyEditor.initialize*
        """
        #
        # create short cut links to appended widgets
        self._pop_name = self.findChild(QLineEdit, 'pop_name')
        self._pop_size = self.findChild(QLineEdit, 'pop_size')
        self._neur_type = self.findChild(QComboBox, 'neur_type')
        
        self._neur_type.addItems(['None'] + self._rep.get_entries('neuron'))
        
    @pyqtSlot(int)
    def update_population(self, pop_id):
        """
        Set the informations of the selected population.
        
        Emitted by:
        
        *stackedWiget_2.signal_update_population*
        """
        self._pop_id = pop_id
        try:
            # in our data storage?
            self._population_data[pop_id] 
        except KeyError:
            # get the data from repository
            obj = self._rep.get_object('network', 'Bar_Learning')[pop_id]
            self._population_data.update( { pop_id : { 'name': obj['name'], 'geometry' : obj['geometry'], 'type': obj['type'] } } ) 
            
        finally:
            self._pop_name.setText(str(self._population_data[pop_id]['name']))
            self._pop_size.setText(str(self._population_data[pop_id]['geometry']))
            
            self._neur_type.setCurrentIndex((['None']+self._rep.get_entries('neuron')).index(self._population_data[pop_id]['type']))

    @pyqtSlot(QString)
    def neur_type_changed(self, type):
        """
        QComboBox selection was changed.
        
        Hint: once called on initialization phase, where is no data available.
        """
        type = str(self._neur_type.currentText())
        
        if self._pop_id != -1:
            # update local data
            self._population_data[self._pop_id]['type'] = type

            # update repository
            self._rep.get_object('network', 'Bar_Learning')[self._pop_id]['type'] = type
            
    @pyqtSlot()
    def pop_name_changed(self):
        """
        QLineEdit was modfied and editFinished was emitted.
        """
        name = self._pop_name.text()
        if self._pop_id != -1:
            # update local data
            self._population_data[self._pop_id]['name'] = name

            # update repository
            self._rep.get_object('network', 'Bar_Learning')[self._pop_id]['name'] = name

    @pyqtSlot()
    def pop_geo_changed(self):
        """
        QLineEdit was modfied and editFinished was emitted.
        """
        try:
            geo = eval(str(self._pop_size.text()))
            
            if self._pop_id != -1:
                # update local data
                self._population_data[self._pop_id]['geometry'] = geo
    
                # update repository
                self._rep.get_object('network', 'Bar_Learning')[self._pop_id]['geometry'] = geo
        except:
            print 'Invalid input for edit field'