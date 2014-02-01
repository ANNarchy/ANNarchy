from .GLObjects import GLBaseWidget, Point2d, Quad2d, Line2d
from PyQt4.QtCore import pyqtSignal, pyqtSlot, QString

# PyOpenGL imports
import OpenGL.GL as gl
import copy

class GLNetworkWidget(GLBaseWidget):
    update_population = pyqtSignal(int, int)
    """
    Main class for visualization of network.
    """    
    def __init__(self, parent=None):
        """
        Constructor.
        
        initializes GL widget and all the basic stuff needed.
        """
        super(GLNetworkWidget, self).__init__(parent)
        
        self.populations = {}
        self.projections = {};
        
        self._quad = None
        self._drawing = False
        self._selected = -1

    def set_repository(self, repo):
        self._rep = repo

    @pyqtSlot(QString)
    def show_network(self, name):
        """
        """
        data = self._rep.get_object('network', name)
        
        # load all data for selected network
        for id, pop in data.iteritems():            
            points = [ Point2d(p[0],p[1]) for p in pop['coords'] ]
            quad = Quad2d().from_p(points[0],
                                   points[1],
                                   points[2],
                                   points[3])
            
            self.populations.update( { id : quad } )
            
        self.updateGL()

    def paintGL(self):
        """
        Overloaded PyQt function.
        
        Rendering the current scene. In contrast to common GL window implementations, 
        e. g. GLut, the PyQt OpenGL has no destinct rendering loop. This function is emitted
        through an updateGL() call.
        
        This function is called explicitly through:
        
        * *resize event*, *mouseMoveEvent*, *VisualizerWidget.render_data*
        """
        # clear the buffer
        gl.glClear(gl.GL_COLOR_BUFFER_BIT) 
        
        gl.glColor3f(0.0,0.0,0.0)
        for id, quad in self.populations.iteritems():
            if id == self._selected:
                gl.glColor3f(1.0,0.0,0.0)
                
            gl.glBegin(gl.GL_LINE_LOOP)
            gl.glVertex2f(quad.p1._x, quad.p1._y)
            gl.glVertex2f(quad.p2._x, quad.p2._y)
            gl.glVertex2f(quad.p3._x, quad.p3._y)
            gl.glVertex2f(quad.p4._x, quad.p4._y)
            gl.glEnd()

            if id == self._selected:
                gl.glColor3f(0.0,0.0,0.0)
                

        for line in self.projections:
            gl.glBegin(gl.GL_LINES)
            gl.glVertex2f(line.p1._x, line.p1._y)
            gl.glVertex2f(line.p2._x, line.p2._y)
            gl.glEnd()

        if self._quad != None:
            gl.glBegin(gl.GL_LINE_LOOP)
            gl.glVertex2f(self._quad.p1._x, self._quad.p1._y)
            gl.glVertex2f(self._quad.p2._x, self._quad.p2._y)
            gl.glVertex2f(self._quad.p3._x, self._quad.p3._y)
            gl.glVertex2f(self._quad.p4._x, self._quad.p4._y)
            gl.glEnd()
    
    def mousePressEvent(self, event):
        """
        Overloaded PyQt function.
        
        This event function is emitted if a mouse button is pressed, we differ two functions, depending on
        the sequel action:
        
        * *release*: the user want to select a population
        * *move*: the user want to draw a new population
        """
        mousePos = event.pos()
        
        self._start_x = mousePos.x()/float(self.width)
        self._start_y = 1.0 - mousePos.y()/float(self.height)
        
        #
        # the mouse and view coord system are invers to each other
        p = Point2d(mousePos.x()/float(self.width), 1.0 - mousePos.y()/float(self.height))
        
        self._selected = -1
        for id, quad in self.populations.iteritems():
            if quad.point_within(p):
                print 'selected quad', id
                self.update_population.emit(1, id)
                self._selected = id
        
        if self._selected == -1: 
            self.update_population.emit(0, 0)
            self._quad = None
            
        self.updateGL()

    def mouseMoveEvent(self, event):
        """
        Overloaded PyQt function.

        While the user presses the mouse key and moves around, we update the visualization. We draw
        a quad with the left buttom corner (where he clicked) and the right upper corner (current mouse
        position)
        """
        self._drawing = True
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
        """
        Overloaded PyQt function.

        We distinguish between a draw attempt and a select by the occurance of a mouseMoveEvent. Through several
        issues it occurs, that a mouse click although emit a minimal mouse movement. To avoid a creation of minimal
        quads we define a minimal size a quad need to have. 
        
        0.1 percent of the visible screen need to be covered by the new quad, else it is ignored. 
        """
        mousePos = event.pos()
        
        try:
            if self._drawing:
                #TODO: maybe a certain percentage of view field
                if self._quad.comp_area() > 0.001: 
                    pop_id = len(self.populations)
                    self.populations.update({ pop_id: copy.deepcopy(self._quad)})
            
                    # create a new entry in repository
                    self._rep.get_object('network','Bar_Learning').update( { pop_id: { 'name': 'Population'+str(pop_id), 'geometry': (1,1), 'coords': copy.deepcopy(self._quad) } } )
                    
            
            self._quad = None
            self._drawing = False
            
        except AttributeError:
            pass # no real quad
        
        self.updateGL()