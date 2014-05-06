from ANNarchy.core.Population import Population
from ANNarchy.core.Neuron import RateNeuron
import ANNarchy.core.Global as Global

try:
    from PIL import Image
except:
    Global._error('The Python Image Library (pillow) is not installed on your system, unable to create ImagePopulations.')
    exit(0)
try:
    from . import CameraDevice as Cam
    exists_cv = True
except Exception, e :
    print e
    Global._warning('OpenCV is not installed on your system, unable to connect to a camera.')
    exists_cv = False
            
import numpy as np

class ImagePopulation(Population):
    """ 
    Specific rate-coded Population allowing to represent images (png, jpg...) or webcam streams as the firing rate of a population (each neuron represents one pixel).
    
    This extension requires the Python Image Library (pip install Pillow) and (only for camera inputs) OpenCV >= 2.0 (apt-get/yum install opencv).
    
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
        if len(geometry)==3 and geometry[2]==1:
            geometry = (geometry[0], geometry[1])
            
        # Create the population     
        Population.__init__(self, geometry = geometry, name=name, neuron = RateNeuron(parameters="""rate = 0.0""") )
        
            
        # Default camera
        self.cam = None

    def __del__(self):
        self.stop_camera()
        
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
            
    def start_camera(self, camera_port=0):
        """
        Starts the webcam with the corresponding device (default = 0).
        
        On linux, the camera port corresponds to the number in /dev/video0, /dev/video1, etc.
        
        The camera must be released at the end of the script:
        
        >>> pop.stop_camera()
        """
        if not exists_cv:
            return
        if self.cam: # The camera is already started
            self.stop_camera()
        try:
            self.cam = Cam.CameraDevice(
                self,
                id=camera_port, 
                width=self.geometry[1], 
                height=self.geometry[0], 
                depth=(self.geometry[2] if self._dimension == 3 else 1)
            )
        except Exception, e:
            print e
            Global._error('The camera ' + str(camera_port) + ' is not available.')
            return
        
    def grab_image(self):
        """
        Grabs one image from the camera and feeds it into the population.
        
        The camera must be first started with:
        
        >>> pop.start_camera(0)
        """
        if not exists_cv:
            return
        if self.cam and self.cam.is_opened():                   
            self.cam.grab_image()
        else:
            Global._error('The camera is not started yet. Call start_camera(0) first.')
   
    def stop_camera(self):
        """
        Releases the camera.
        """
        if not exists_cv:
            return
        if self.cam:
            self.cam.release()
            self.cam=None
