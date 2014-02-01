# PyQT4 imports
from PyQt4 import QtGui, QtCore, QtOpenGL
from PyQt4.QtOpenGL import QGLWidget
# PyOpenGL imports
import OpenGL.GL as gl
import math

class Point2d(object):
    def __init__(self,x,y):
        self._x = x
        self._y = y 
        
    def __repr__(self):
        return '[ '+str(self._x)+', '+str(self._y)+' ]'
    
    def __add__(self, other):
        res = Point2d(0.0,0.0)
        if isinstance(other, Point2d):
            res._x = self._x + other._x
            res._y = self._y + other._y
        else: #scalar
            res._x = self._x + other
            res._y = self._y + other
        return res

    def __sub__(self, other):
        res = Point2d(0.0,0.0)
        if isinstance(other, Point2d):
            res._x = self._x - other._x
            res._y = self._y - other._y
        else: #scalar
            res._x = self._x - other
            res._y = self._y - other
        return res
    
    def __mul__(self, other):
        res = Point2d(0.0,0.0)
        if isinstance(other, Point2d):
            res._x = self._x * other._x
            res._y = self._y * other._y
        else: #scalar
            res._x = self._x * other
            res._y = self._y * other
        return res

    def __div__(self, other):
        res = Point2d(0.0,0.0)
        if isinstance(other, Point2d):
            res._x = self._x / other._x
            res._y = self._y / other._y
        else: #scalar
            res._x = self._x / other
            res._y = self._y / other
        return res
    
    @property
    def length(self):
        return math.sqrt( (self._x * self._x) + (self._y * self._y) ) 
                    
class Quad2d(object):
    def __init__(self, center=Point2d(0,0), radius=0):
        if radius < 0:
            radius = (-1.0)* radius
            
        self.p1 = center - Point2d( radius,  radius)
        self.p2 = center - Point2d(-radius,  radius)
        self.p3 = center - Point2d(-radius, -radius)
        self.p4 = center - Point2d( radius, -radius)

    def from_p(self, p1, p2, p3, p4):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        return self
    
    def __repr__(self):
        return  '[ '+'[ '+str(self.p1._x)+', '+str(self.p1._y)+' ] [ '+str(self.p2._x)+', '+str(self.p2._y)+' ] [ '+str(self.p3._x)+', '+str(self.p3._y)+' ] [ '+str(self.p4._x)+', '+str(self.p4._y)+' ] ]'
                
    def point_within(self, p):
        
        if (p._x < self.p1._x) or (p._x > self.p2._x):
            #print ('\tx_test failed')
            return False;

        if (p._y < self.p3._y) or (p._y > self.p1._y):
            #print ('\ty_test failed')
            return False;
        
        return True;
    
    def comp_area(self):
        # we assume a 3dimensional vector and the 3rd component is 1
        return ((self.p2 - self.p1).length) * ((self.p2 - self.p3).length)
        
class Line2d(object):
    def __init__(self, p1, p2):
        self.p1 = p1 
        self.p2 = p2
        
    @property
    def length(self):
        return (self.p2 - self.p1).length

class GLBaseWidget(QGLWidget):
    """
    Base object for all GL based widgets in ANNarchy4
    """
    def __init__(self, parent):
        super(GLBaseWidget, self).__init__(parent)
        
        self.initializeGL()
        self.width = 600
        self.height = 600
        
    def initializeGL(self):
        """
        Initialize OpenGL, VBOs, upload data on the GPU, etc.
        """
        # background color
        gl.glClearColor(1.0,1.0,1.0,0) # background color
        gl.glDisable(gl.GL_DEPTH_TEST) # we don't need depth test in 2D case
        gl.glLineWidth(3.0)

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
        gl.glOrtho(0, 1, 0, 1, 0, 1)
