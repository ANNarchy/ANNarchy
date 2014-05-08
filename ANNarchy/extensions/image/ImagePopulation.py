from ANNarchy.core.Population import Population
from ANNarchy.core.Neuron import RateNeuron
import ANNarchy.core.Global as Global

try:
    from PIL import Image
except:
    Global._error('The Python Image Library (pillow) is not installed on your system, unable to create ImagePopulations.')
    exit(0)
            
import numpy as np

class ImagePopulation(Population):
    """ 
    Specific rate-coded Population allowing to represent images (png, jpg...) as the firing rate of a population (each neuron represents one pixel).
    
    This extension requires the Python Image Library (pip install Pillow).
    
    Usage:
    
    >>> from ANNarchy import *
    >>> from ANNarchy.extensions.image import ImagePopulation
    >>> pop = ImagePopulation(name='Input', geometry=(480, 640))
    >>> pop.set_image('image.jpg')
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
            self.cyInstance._set_rate(np.array(im)/255.)


class VideoPopulation(ImagePopulation):
    """ 
    Specific rate-coded Population allowing to feed a webcam input into the firing rate of a population (each neuron represents one pixel).
    
    This extension requires the C++ library OpenCV >= 2.0 (apt-get/yum install opencv).
    
    Usage :
    
    >>> from ANNarchy import *
    >>> from ANNarchy.extensions.image import VideoPopulation
    >>> pop = VideoPopulation(name='Input', geometry=(480, 640))
    >>> compile()
    >>> pop.start_camera(0)
    >>> while(True):
    ...   pop.grab_image()
    ...   simulate(10.0)
    """
    
    def __init__(self, geometry, name=None):
        """        
        *Parameters*:
        
            * *geometry*: population geometry as tuple. It must be fixed through the whole simulation. If the camera provides images of a different size, it will be resized.
            
                * If the geometry is 2D, it corresponds to the (height, width) of the image. Only the luminance of the pixels will be represented (grayscale image).
            
                * If the geometry is 3D, the third dimension can be either 1 (grayscale) or 3 (color).
                
                If the third dimension is 3, each will correspond to the RGB values of the pixels.
                
                .. warning::
                
                    Due to the indexing system of Numpy, a 640*480 image should be fed into a (480, 640) or (480, 640, 3) population.

            * *name*: unique name of the population (optional).
        
        """         
        # Create the population     
        ImagePopulation.__init__(self, geometry = geometry, name=name )

        # Code generator
        from .VideoPopulationGenerator import VideoPopulationGenerator
        self.generator = VideoPopulationGenerator(self)
            
    def start_camera(self, camera_port=0):
        """
        Starts the webcam with the corresponding device (default = 0).
        
        On linux, the camera port corresponds to the number in /dev/video0, /dev/video1, etc.
        """
        self.cyInstance.start_camera(camera_port, self.geometry[1], self.geometry[0], 3 if self._dimension==3 else 1)
        
    def grab_image(self):
        """
        Grabs one image from the camera and feeds it into the population.
        
        The camera must be first started with:
        
        >>> pop.start_camera(0)
        """
        self.cyInstance.grab_image()

        
