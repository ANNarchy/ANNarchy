import os
import re

from PyQt4.QtGui import QFileDialog
from PyQt4.Qsci import QsciScintilla, QsciLexerPython, QsciLexerXML
from PyQt4 import QtCore

from CodeTemplates import *

class CodeView(QsciScintilla):

    def __init__(self):
        QsciScintilla.__init__(self)
        self._loaded_script = None
        
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
    
    @QtCore.pyqtSlot(int, str)
    def set_code(self, type, name):
        
        if type == 0:
            self.setLexer(QsciLexerPython(self))
            code = rate_neuron_def % { 'name': name }
            self.setText(code)
        else:
            print 'unknown type.'

    def save(self):
        
        self.setModified(True)

    def check_changes(self):
        return self.isModified()