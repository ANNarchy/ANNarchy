from .GLObjects import GLBaseWidget, Point2d, Quad2d, Line2d

# PyOpenGL imports
import OpenGL.GL as gl
import copy

class GLNetworkWidget(GLBaseWidget):
    """
    Main class for visualization of network.
    """    
    def __init__(self, parent, main_window):
        super(GLNetworkWidget, self).__init__(parent)
        
        self._main_window = main_window
        self.populations = {}
        self.populations.update( { 0 : Quad2d(Point2d(0.5,0.5), 0.05) } )
        self.populations.update( { 1 : Quad2d(Point2d(0.3,0.5), 0.05) } )
        
        self.projections = [];
        self.projections.append(Line2d(Point2d(0.5,0.5), Point2d(0.3,0.5)))
        
        self._quad = None

    def paintGL(self):
        """
        Paint the scene.
        """
        # clear the buffer
        gl.glClear(gl.GL_COLOR_BUFFER_BIT) 
        
        gl.glColor3f(0.0,0.0,0.0)
        for quad in self.populations.itervalues():

            gl.glBegin(gl.GL_LINE_LOOP)
            gl.glVertex2f(quad.p1._x, quad.p1._y)
            gl.glVertex2f(quad.p2._x, quad.p2._y)
            gl.glVertex2f(quad.p3._x, quad.p3._y)
            gl.glVertex2f(quad.p4._x, quad.p4._y)
            gl.glEnd()

        #gl.glColor3f(0.0,0.6,0.0)
        for line in self.projections:

            gl.glBegin(gl.GL_LINES)
            gl.glVertex2f(line.p1._x, line.p1._y)
            gl.glVertex2f(line.p2._x, line.p2._y)
            gl.glEnd()

        if self._quad != None:
            #gl.glColor3f(0.0,0.0,0.0)
            gl.glBegin(gl.GL_LINE_LOOP)
            gl.glVertex2f(self._quad.p1._x, self._quad.p1._y)
            gl.glVertex2f(self._quad.p2._x, self._quad.p2._y)
            gl.glVertex2f(self._quad.p3._x, self._quad.p3._y)
            gl.glVertex2f(self._quad.p4._x, self._quad.p4._y)
            gl.glEnd()
    
    def mousePressEvent(self, event):
        mousePos = event.pos()
        print 'start', mousePos
        
        self._start_x = mousePos.x()/float(self.width)
        self._start_y = 1.0 - mousePos.y()/float(self.height)
        
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
            self._quad = None
            
        self.updateGL()

    def mouseMoveEvent(self, event):
        mousePos = event.pos()
        
        self._stop_x = float(mousePos.x()/float(self.width))
        self._stop_y = 1.0 - float(mousePos.y()/float(self.height))
        
        self._quad = Quad2d().from_p(
                            Point2d(self._start_x, self._start_y),
                            Point2d(self._stop_x, self._start_y),
                            Point2d(self._stop_x, self._stop_y),
                            Point2d(self._start_x, self._stop_y)                            
                            )
        
        self.updateGL()
        #print mousePos
        
    def mouseReleaseEvent(self, event):
        mousePos = event.pos()
        
        try:
            #TODO: maybe a certain percentage of view field
            if self._quad.comp_area() > 0.001: 
                self.populations.update({ len(self.populations): copy.deepcopy(self._quad)})
        except AttributeError:
            pass # no real quad
        
        self.updateGL()