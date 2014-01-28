from .GLObjects import GLBaseWidget, Point2d, Quad2d, Line2d

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
