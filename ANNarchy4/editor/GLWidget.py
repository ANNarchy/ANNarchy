# PyQT4 imports
from PyQt4 import QtGui, QtCore, QtOpenGL
from PyQt4.QtOpenGL import QGLWidget
# PyOpenGL imports
import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo
import numpy as np
import numpy.random as rdn
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
            res._x = self._y + other
        return res

    def __sub__(self, other):
        res = Point2d(0.0,0.0)
        if isinstance(other, Point2d):
            res._x = self._x - other._x
            res._y = self._y - other._y
        else: #scalar
            res._x = self._x - other
            res._x = self._y - other
        return res
        
class Quad(object):
    def __init__(self, center=Point2d(0,0), radius=0):
        if radius < 0:
            radius = (-1.0)* radius
            
        self.p1 = center - Point2d( radius,  radius)
        self.p2 = center - Point2d(-radius,  radius)
        self.p3 = center - Point2d(-radius, -radius)
        self.p4 = center - Point2d( radius, -radius)
        
        print self

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

        if (p._y < self.p1._y) or (p._y > self.p3._y):
            #print ('\ty_test failed')
            return False;
        
        return True;
        
class Line(object):
    def __init__(self, p1, p2):
        self.p1 = p1 
        self.p2 = p2
            
class GLWidget(QGLWidget):
    
    def __init__(self, parent):
        super(GLWidget, self).__init__(parent)
        
        self.initializeGL()
        self.width = 600
        self.height = 600
        
    def initializeGL(self):
        """
        Initialize OpenGL, VBOs, upload data on the GPU, etc.
        """
        # background color
        gl.glClearColor(0.9,0.9,0.9,0) # background color
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
                    
class VisualizerGLWidget(GLWidget):
    _update_grid = QtCore.pyqtSignal(int, int)
        
    def __init__(self, parent):
        super(VisualizerGLWidget, self).__init__(parent)
        self._offset = 0.0
        
        self._update_grid.connect(self.update_grid)
        
        self._grid = { 0: Quad(Point2d(0.5,0.5), 0.5) }
        self._x_dim = 1
        self._y_dim = 1
        
    @QtCore.pyqtSlot(int, int)    
    def update_grid(self, num_x, num_y):
        g_w = 1.0 / float(num_x)
        g_h = 1.0 / float(num_y)
        
        self._x_dim = num_x
        self._x_dim = num_y
        
        print g_w, g_h
        self._grid = {}
        
        if num_x == 1 and num_y == 1:
            q = Quad().from_p(
                         Point2d(0.0, 0.0),
                         Point2d(1.0, 0.0),
                         Point2d(1.0, 1.0),
                         Point2d(0.0, 1.0)
                         )
            self._grid.update( { 0 : q } )
        else:
            for y in xrange(num_y):
                for x in xrange(num_x):
                    q = Quad().from_p(
                             Point2d(x * g_w, y * g_h),
                             Point2d((x+1) * g_w, y * g_h),
                             Point2d((x+1) * g_w, (y+1) * g_h),
                             Point2d(x * g_w, (y+1) * g_h)
                             )
                    id = y * self._x_dim + x
                    
                    print q
                    self._grid.update( { id : q } )
                
        self.updateGL()
        
    def paintGL(self):
        """
        Paint the scene.
        """
        # clear the buffer
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
        gl.glColor3f(0,0,0)
        for quad in self._grid.itervalues():

            gl.glBegin(gl.GL_LINE_LOOP)
            gl.glVertex2f(quad.p1._x, quad.p1._y)
            gl.glVertex2f(quad.p2._x, quad.p2._y)
            gl.glVertex2f(quad.p3._x, quad.p3._y)
            gl.glVertex2f(quad.p4._x, quad.p4._y)
            gl.glEnd()
        
    def mousePressEvent(self, event):
        mousePos = event.pos()
        
        p = Point2d(mousePos.x()/float(self.width), mousePos.y()/float(self.height))
        
        for id, quad in self._grid.iteritems():
            if quad.point_within(p):
                print 'selected a cell', id
        
#===============================================================================
#         gl.glPointSize(3.0)
#         self._offset += 1.0
#         self._offset = self._offset % 360
# 
#         x = [ i for i in range(360)]
#         y = [ math.cos( float(i+self._offset) * math.pi / 180.0) for i in range(360) ]
#         y2 = [ math.sin( float(i+self._offset) * math.pi / 180.0) for i in range(360) ]
#         
#         gl.glColor3f(0.0,0.0,0.6)                
#         gl.glBegin(gl.GL_LINE_STRIP)
#         for i in range(360):
#              gl.glVertex2f(x[i], y[i])
#         gl.glEnd()
#         gl.glColor3f(0.0,0.6,0.0)                
#         gl.glBegin(gl.GL_LINE_STRIP)
#         for i in range(360):
#              gl.glVertex2f(x[i], y2[i])
#         gl.glEnd()
#===============================================================================

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
        
class NetworkGLWidget(GLWidget):
    
    def __init__(self, parent, main_window):
        super(NetworkGLWidget, self).__init__(parent)
        
        self._main_window = main_window
        self.populations = {}
        self.populations.update( { 0 : Quad(Point2d(0.5,0.5), 0.05) } )
        self.populations.update( { 1 : Quad(Point2d(0.3,0.5), 0.05) } )
        
        self.projections = [];
        self.projections.append(Line(Point2d(0.5,0.5), Point2d(0.3,0.5)))

    def paintGL(self):
        """
        Paint the scene.
        """
        # clear the buffer
        gl.glClear(gl.GL_COLOR_BUFFER_BIT) 
        
        gl.glColor3f(0.6,0.0,0.0)
        for quad in self.populations.itervalues():

            gl.glBegin(gl.GL_LINE_LOOP)
            gl.glVertex2f(quad.p1._x, quad.p1._y)
            gl.glVertex2f(quad.p2._x, quad.p2._y)
            gl.glVertex2f(quad.p3._x, quad.p3._y)
            gl.glVertex2f(quad.p4._x, quad.p4._y)
            gl.glEnd()

        gl.glColor3f(0.0,0.6,0.0)
        for line in self.projections:

            gl.glBegin(gl.GL_LINES)
            gl.glVertex2f(line.p1._x, line.p1._y)
            gl.glVertex2f(line.p2._x, line.p2._y)
            gl.glEnd()
    
    def mousePressEvent(self, event):
        mousePos = event.pos()
        
        #
        # the mouse and view coord system are invers to each other
        p = Point2d(mousePos.x()/float(self.width), 1.0 - mousePos.y()/float(self.height))
        
        selected = False
        for id, quad in self.populations.iteritems():
            if quad.point_within(p):
                self._main_window.signal_net_editor_to_pop_view.emit(1, id) # 1 = population view
                selected = True
        
        if not selected: 
            self._main_window.signal_net_editor_to_pop_view.emit(0, 0) # 0 = population view
        