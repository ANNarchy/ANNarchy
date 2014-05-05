from ANNarchy.core.Population import Population
from ANNarchy.core.Neuron import RateNeuron
import ANNarchy.core.Global as Global

try:
    from PIL import Image
except:
    Global._error('The Python Image Library (pillow) is not installed on your system, unable to create ImagePopulations.')
    exit(0)
try:
    import cv2
    exist_cv = True
except:
    Global._warning('The Python bindings to OpenCV are not installed on your system, unable to connect to a camera.')
    exist_cv = False
            
import atexit
import numpy as np

class ImagePopulation(Population):
    """ 
    Specific rate-coded Population allowing to represent images (png, jpg...) or webcam streams as the firing rate of a population (each neuron represents one pixel).
    
    This extension requires the Python Image Library (pip install Pillow) and (only for camera inputs) OpenCV >= 2.0 (apt-get/yum install python-opencv).
    
    Usage for images:
    
    >>> from ANNarchy import *
    >>> from ANNarchy.extensions.image import ImagePopulation
    >>> pop = ImagePopulation(name='Input', geometry=(480, 640))
    >>> pop.set_image('image.jpg')
    
    Usage for cameras inputs:
    
    >>> from ANNarchy import *
    >>> from ANNarchy.extensions.image import ImagePopulation
    >>> pop = ImagePopulation(name='Input', geometry=(480, 640))
    >>> compile()
    >>> pop.start_camera(0)
    >>> pop.grab_image()
    >>> pop.stop_camera()
    """
    
    def __init__(self, geometry, name=None):
        """        
        *Parameters*:
        
            * *geometry*: population geometry as tuple. It must correspond to the image size and be fixed through the whole simulation.
            
                * If the geometry is 2D, it corresponds to the (height, width) of the image. Only the luminance of the pixels will be represented (grayscale image).
            
                * If the geometry is 3D, the third dimension can be either 1 (grayscale) or 3 (color).
                
                If the third dimension is 3, each will correspond to the RGB values of the pixels.
                
                .. warning::
                
                    Due to the indexing system of Numpy, a 640*480 image should be fed into a (480, 640) or (480, 640, 3) population.

            * *name*: unique name of the population (optional).
        
        """   
        # Check geometry
        if isinstance(geometry, int) or len(geometry)==1:
            Global._error('The geometry of an ImagePopulation should be 2D (grayscale) or 3D (color).')
            exit(0)
        if len(geometry)==3 and (geometry[2]!=3 and geometry[2]!=1):
            Global._error('The third dimension of an ImagePopulation should be either 1 (grayscale) or 3 (color).') 
            exit(0)
            
        # Create the population     
        Population.__init__(self, geometry = geometry, name=name, neuron = RateNeuron(parameters="""rate = 0.0""") )
        
        if len(geometry)==3 and geometry[2]==1:
            self.dimension = 1
            
        # Default camera
        self.cam = None

        
    def set_image(self, image_name):
        """ 
        Sets an image (.png, .jpg or whatever is supported by PIL) into the firing rate of the population.
        
        If the image has a different size from the population, it will be resized.
        
        """
        try:
            im = Image.open(image_name)
        except : # image does not exist
            Global._error('The image ' + image_name + ' does not exist.')
            exit(0)
        # Resize the image if needed
        (width, height) = (self.geometry[1], self.geometry[0])
        if im.size != (width, height):
            Global._warning('The image ' + image_name + ' does not have the same size '+str(im.size)+' as the population ' + str((width, height)) + '. It will be resized.')
            im = im.resize((width, height))
        # Check if only the luminance should be extracted
        if self.dimension == 2 or self.geometry[2] == 1:
            im=im.convert("L")
        # Set the rate of the population
        if not Global._compiled:
            self.rate = (np.array(im))/255.
        else:
            setattr(self.cyInstance, 'rate', (np.array(im))/255.)
            
    def start_camera(self, id_camera=0):
        """
        Starts the webcam with the corresponding device (default = 0).
        
        The camera must be released at the end of the script:
        
        >>> pop.stop_camera()
        """
        if self.cam: # The camera is already started
            self.stop_camera()
        try:
            self.cam = cv2.VideoCapture(id_camera)
            self.cam.set(cv2.CV_CAP_PROP_FRAME_WIDTH,self.geometry[1]) # sets the width
            self.cam.set(cv2.CV_CAP_PROP_FRAME_HEIGHT,self.geometry[0]) # sets the height
            self.cam.set(cv2.CV_CAP_PROP_CONVERT_RGB, True) # sets the color space
        except:
            Global._error('The camera ' + str(id_camera) + ' is not available.')
            return
        
    def grab_image(self):
        """
        Grabs one image from the camera and feeds it into the population.
        
        The camera must be first started with:
        
        >>> pop.start_camera(0)
        """
        if self.cam:
            ret, im = self.cam.read()
            if ret:
                if self.dimension == 2 or self.geometry[2] == 1:
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                else:
                    b,g,r = cv2.split(im)       # get b,g,r
                    im = cv2.merge([r,g,b])     # switch it to rgb
                if not Global._compiled:
                    self.rate = (cv2.resize(im, (self.geometry[1], self.geometry[0])))/255.
                else:
                    setattr(self.cyInstance, 'rate', (cv2.resize(im, (self.geometry[1], self.geometry[0])))/255.)
        else:
            Global._error('The camera is not started yet. Call start_camera(0) first.')

    @atexit.register    
    def stop_camera(self):
        """
        Releases the camera.
        """
        if self.cam:
            self.cam.release()
            self.cam=None