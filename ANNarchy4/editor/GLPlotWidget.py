# PyQT4 imports
from PyQt4 import QtGui, QtCore, QtOpenGL
from PyQt4.QtOpenGL import QGLWidget

# PyOpenGL imports
import OpenGL.GL as gl

import numpy as np
import numpy.random as rdn
import math

from .GLObjects import GLBaseWidget
          
class GLPlot1d(GLBaseWidget):
    _render = QtCore.pyqtSignal()
        
    def __init__(self, parent):
        super(GLPlot1d, self).__init__(parent)

        self._render.connect(self.render)
        self._x_data = [ x for x in xrange(360)]
        self._y_data = [0.0 for y in xrange(360)]
        self._count = 360
        
        self._color = [ 0.0, 0.0, 0.6 ]

    @QtCore.pyqtSlot()
    def render(self):
        self.updateGL()
        
    def set_color(self, r, g, b):
        self._color = [ r, g, b];

    def set_data(self, x_data, y_data=[]):
        self._y_data = y_data
        self._count = len(y_data)
        if x_data == []:
            self._x_data = [ x for x in xrange(len(y_data))]
        else:
            self._x_data = x_data
        
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        # the window corner OpenGL coordinates are (-+1, -+1)
        gl.glOrtho(0, self._count, -1, 1, 0, 1)
        
    def paintGL(self):
        """
        Paint the scene.
        """
        # clear the buffer
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
        gl.glColor3f(self._color[0],
                     self._color[1],
                     self._color[2]
                     )
                        
        gl.glBegin(gl.GL_LINE_STRIP)
        for i in range(self._count):
             gl.glVertex2f(self._x_data[i], self._y_data[i])
        gl.glEnd()

                                
    def resizeGL(self, width, height):
        """
        Called upon window resizing: reinitialize the viewport.
        """
        # update the window size
        self.width, self.height = width, height
        # paint within the whole window
        gl.glViewport(0, 0, self.width, self.height)
        # set orthographic projection (2D only)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        # the window corner OpenGL coordinates are (-+1, -+1)
        gl.glOrtho(0, self._count, -1, 1, 0, 1)
        
class GLPlot2d(GLBaseWidget):
    _render = QtCore.pyqtSignal()
        
    def __init__(self, parent):
        super(GLPlot1d, self).__init__(parent)

        self._render.connect(self.render)
        self._data = []
        self._x_size = 0
        self._y_size = 0
        
        self._color = [ 0.0, 0.0, 0.6 ]

    @QtCore.pyqtSlot()
    def render(self):
        self.updateGL()
        
    def set_color(self, r, g, b):
        self._color = [ r, g, b];

    def set_data(self, x, y):
        pass
            
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        # the window corner OpenGL coordinates are (-+1, -+1)
        gl.glOrtho(0, 360, -1, 1, 0, 1)
        
    def paintGL(self):
        """
        Paint the scene.
        """
        # clear the buffer
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
    def resizeGL(self, width, height):
        """
        Called upon window resizing: reinitialize the viewport.
        """
        # update the window size
        self.width, self.height = width, height
        # paint within the whole window
        gl.glViewport(0, 0, self.width, self.height)
        # set orthographic projection (2D only)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        # the window corner OpenGL coordinates are (-+1, -+1)
        gl.glOrtho(0, 1, -1, 1, 0, 1)        
                                
class SineCosineDemo(GLBaseWidget):
    _render = QtCore.pyqtSignal()
        
    def __init__(self, parent):
        super(GLPlot1d, self).__init__(parent)

        self._offset = 0.0
        self._render.connect(self.render)
        
    @QtCore.pyqtSlot()
    def render(self):
        self._offset += 1.0
        self._offset = self._offset % 360.0
        self.updateGL()

    def paintGL(self):
        """
        Paint the scene.
        """
        # clear the buffer
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
        gl.glPointSize(3.0)
        self._offset += 1.0
        self._offset = self._offset % 360
 
        x = [ i for i in range(360)]
        y = [ math.cos( float(i+self._offset) * math.pi / 180.0) for i in range(360) ]
        y2 = [ math.sin( float(i+self._offset) * math.pi / 180.0) for i in range(360) ]
         
        gl.glColor3f(0.0,0.0,0.6)                
        gl.glBegin(gl.GL_LINE_STRIP)
        for i in range(360):
             gl.glVertex2f(x[i], y[i])
        gl.glEnd()
        gl.glColor3f(0.0,0.6,0.0)                
        gl.glBegin(gl.GL_LINE_STRIP)
        for i in range(360):
             gl.glVertex2f(x[i], y2[i])
        gl.glEnd()
                                
    def resizeGL(self, width, height):
        """
        Called upon window resizing: reinitialize the viewport.
        """
        # update the window size
        self.width, self.height = width, height
        # paint within the whole window
        gl.glViewport(0, 0, self.width, self.height)
        # set orthographic projection (2D only)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        # the window corner OpenGL coordinates are (-+1, -+1)
        gl.glOrtho(0, 360, -1, 1, 0, 1)        