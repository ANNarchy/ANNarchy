import os
import re

from PyQt4.QtGui import QFileDialog
from PyQt4.Qsci import QsciScintilla, QsciLexerPython, QsciLexerXML

class CodeView(QsciScintilla):
    def __init__(self, main_window, w, h):
        QsciScintilla.__init__(self, main_window)
        self.setMinimumSize(w,h)
        
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
        