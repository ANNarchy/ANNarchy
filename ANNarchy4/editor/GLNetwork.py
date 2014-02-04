from .GLObjects import GLBaseWidget, Point2d, Quad2d, Line2d
from PyQt4.QtCore import pyqtSignal, pyqtSlot, QString

# PyOpenGL imports
import OpenGL.GL as gl
import copy

class GLNetworkWidget(GLBaseWidget):
    update_object = pyqtSignal(QString, int, int)
    
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
        self._line = None
        
        self._drawing_quad = False
        self._drawing_line = False
        self._selected_quad = -1
        self._selected_line = -1
        self._net_name = 'None'

    def set_repository(self, repo):
        self._rep = repo

    @pyqtSlot(QString)
    def show_network(self, name):
        """
        """
        self._net_name = name
        data = self._rep.get_object('network', name)
        
        # load all data for selected network
        for id, pop in data['pop_data'].iteritems():            
            self.populations.update( { id : pop['coords'] } )

        for id, proj in data['proj_data'].iteritems():            
            self.projections.update( { id : (proj['pre'],proj['post']) } )
            
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
            if id == self._selected_quad:
                gl.glColor3f(1.0,0.0,0.0)
                
            gl.glBegin(gl.GL_LINE_LOOP)
            gl.glVertex2f(quad.p1._x, quad.p1._y)
            gl.glVertex2f(quad.p2._x, quad.p2._y)
            gl.glVertex2f(quad.p3._x, quad.p3._y)
            gl.glVertex2f(quad.p4._x, quad.p4._y)
            gl.glEnd()

            if id == self._selected_quad:
                gl.glColor3f(0.0,0.0,0.0)
                

        for id, line in self.projections.iteritems():
            if id == self._selected_line:
                gl.glColor3f(0.0,1.0,0.0)
            
            p1 = self.populations[line[0]].center
            p2 = self.populations[line[1]].center
            
            gl.glBegin(gl.GL_LINES)
            gl.glVertex2f( p1._x, p1._y )
            gl.glVertex2f( p2._x, p2._y )
            gl.glEnd()

            if id == self._selected_line:
                gl.glColor3f(0.0,0.0,0.0)

        #
        # temporary objects
        if self._line != None:
            gl.glBegin(gl.GL_LINES)
            gl.glVertex2f(self._line.p1._x, self._line.p1._y)
            gl.glVertex2f(self._line.p2._x, self._line.p2._y)
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
        * *move*: the user want to draw a new population or connect two population. If the user 
                  starts on blank space the first is assumed else the second.
         
        """
        mousePos = event.pos()
        
        self._start_x = mousePos.x()/float(self.width)
        self._start_y = 1.0 - mousePos.y()/float(self.height)
        
        #
        # the mouse and view coord system are invers to each other
        p = Point2d(mousePos.x()/float(self.width), 1.0 - mousePos.y()/float(self.height))
        
        self._selected_quad = -1
        self._selected_line = -1
        
        for id, line in self.projections.iteritems():
            p1 = self.populations[line[0]].center
            p2 = self.populations[line[1]].center
            
            if Line2d(p1,p2).is_on_line(p):
                self.update_object.emit(self._net_name, 2, id)
                self._selected_line = id

        for id, quad in self.populations.iteritems():
            if quad.point_within(p):
                # update population view
                self.update_object.emit(self._net_name, 1, id)
                
                # user feed back
                self._selected_quad = id
                self._drawing_line = True #line drawing could be only start from quad
                
        if self._selected_quad == -1 and self._selected_line == -1: 
            self.update_object.emit(self._net_name, 0, 0)
            self._quad = None
            self._line = None
            self._drawing_quad = True
            self._drawing_line = False

        self.updateGL()

    def mouseMoveEvent(self, event):
        """
        Overloaded PyQt function.

        While the user presses the mouse key and moves around, we update the visualization. We draw
        a quad with the left buttom corner (where he clicked) and the right upper corner (current mouse
        position)
        """
        mousePos = event.pos()
        
        self._stop_x = float(mousePos.x()/float(self.width))
        self._stop_y = 1.0 - float(mousePos.y()/float(self.height))

        if self._drawing_quad:
            self._quad = Quad2d().from_p(
                                Point2d(self._start_x, self._start_y),
                                Point2d(self._stop_x, self._start_y),
                                Point2d(self._stop_x, self._stop_y),
                                Point2d(self._start_x, self._stop_y)                            
                                )
        elif self._drawing_line:
            self._line = Line2d(Point2d(self._start_x, self._start_y), Point2d(self._stop_x, self._stop_y))
            
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
            if self._drawing_quad:
                #TODO: maybe a certain percentage of view field
                if self._quad.comp_area() > 0.001: 
                    pop_id = len(self.populations)
                    self.populations.update( { pop_id: copy.deepcopy(self._quad)} )
            
                    # create a new entry in repository
                    pop_obj = {
                              'name': 'Population'+str(pop_id), 
                              'geometry': (1,1), 
                              'coords': copy.deepcopy(self._quad),
                              'type' : 'None' 
                              }
                    self._rep.get_object('network','Bar_Learning')['pop_data'].update( { pop_id: pop_obj } )
                
                self._drawing_quad = False
                self._quad = None
                
            if self._drawing_line:
                if self._line.length > 0.001:
                    for id, quad in self.populations.iteritems():
                        if quad.point_within(self._line.p2):

                            proj_id = len(self.projections)
                            self.projections.update( { proj_id : (self._selected_quad, id) } )
                            import pprint
                            pprint.pprint(self.projections)

                            proj_obj = {
                                      'pre': self._selected_quad, 
                                      'post': id,
                                      'target': 'None' 
                                      }
                            self._rep.get_object('network','Bar_Learning')['proj_data'].update( { proj_id: proj_obj } )

            
                self._drawing_line = False
                self._line = None
                
            
        except AttributeError:
            pass # no real quad
        
        self.updateGL()