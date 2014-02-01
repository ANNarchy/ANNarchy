"""
    
    CodeView.py
    
    This file is part of ANNarchy.
    
    Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
    Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ANNarchy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
"""
import os

from PyQt4.QtGui import QFileDialog, QMessageBox
from PyQt4.Qsci import QsciScintilla, QsciLexerPython, QsciLexerXML
from PyQt4.QtCore import pyqtSlot, pyqtSignal, QString 

from CodeTemplates import *

class CodeView(QsciScintilla):
    
    def __init__(self, parent):
        QsciScintilla.__init__(self, parent)
        self._loaded_script = None
        self._curr_name = None
        self._curr_type = None

        self.setLexer(QsciLexerPython(self))

    def set_repository(self, repo=None):
        self._rep = repo
        print repo
        
    @pyqtSlot()
    def initialize(self):
        
        if len(self._rep.get_entries('neuron'))>0:
            self._curr_name = self._rep.get_entries('neuron')[0]
            self._curr_type = 'neuron'
            self.setText(self._rep.get_object('neuron', self._curr_name))

    @pyqtSlot()
    def save(self):
        print 'xxx'
        self._rep.update( self._curr_name, self.text() )
    
    @pyqtSlot(int, QString)
    def set_code(self, type, name):
        if type == 0 :
            obj = 'neuron'
        elif type == 2:
            obj = 'synapse'

        #
        # create the new entry
        if not self._rep.entry_contained(name):
            if type == 0 :
                code = rate_neuron_def % { 'name': name }
            elif type == 2:
                code = rate_synapse_def % { 'name': name }
                
            self._rep.add_object(obj, name, code)
            if self._curr_name == None:
                self.setText(code)

        #
        # check if an other definition is remain                
        if self._rep.entry_contained(self._curr_name):
            if self._rep.get_object(self._curr_type, self._curr_name) != self.text():
                reply = QMessageBox.question(self, 'Save changes', 'You want to save the modified object?', QMessageBox.Yes, QMessageBox.No )
             
                #
                # ask the user if he want to save   
                if reply == QMessageBox.Yes:
                    self._rep.update_object(self._curr_type, self._curr_name, self.text())
    
            # show the new data set
            self.setText( self._rep.get_object( obj, name ) )
            
        self._curr_name = name
        self._curr_type = obj

    def load_file(self):
        filename = QFileDialog.getOpenFileName(self, 'Open File', os.getcwd())
        
        self._loaded_script = filename
        fileName, fileExtension = os.path.splitext(str(filename))
        
        if fileExtension == ".xml":
            self.setLexer(QsciLexerXML(self))
            self.setText(open(filename).read())
        else:
            self.setLexer(QsciLexerPython(self))
            self.setText(open(filename).read())
            
    @property
    def loaded_script(self):
        return self._loaded_script
