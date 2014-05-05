from ANNarchy.parser.Analyser import analyse_population
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
    Specific Population allowing to represent images (png, jpg...).
    
    """
    
    def __init__(self, geometry, name=None):
        """        
        *Parameters*:
        
            * *geometry*: population geometry as tuple. It must correspond to the image size and be fixed through the whole simulation.
            
                * If the geometry is 2D, it corresponds to the (height, width) of the image. Only the luminance of the pixels will be represented (grayscale image).
            
                * If the geometry is 3D, the third dimension must be 3 and corresponds to the RGB values of each pixel.
                
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

        
         
        
        
    def set_image(self, image_name):
        """ 
        Sets an image (.png, .jpg or whatever is supported by PIL) into the rate of the population.
        
        If the image has a different size from the population, it will be resized.
        
        """
        try:
            im = Image.open(image_name)
        except : # image does not exist
            Global._error('The image ' + image_name + ' does not exist.')
            exit(0)
        # Resize the image if needed
        (width, height) = (self.geometry[1], self.geometry[0])
        if im.size != self.geometry[:2]:
            Global._warning('The image ' + image_name + ' does not have the same size '+str(im.size)+' as the population ' + str((width, height)) + '. It will be resized.')
            im = im.resize((width, height))
        # Check if only the luminance should be extracted
        if self.dimension == 2 or self.geometry[2] == 1:
            im=im.convert("L")
        # Set the rate of the population
        self.rate = (np.array(im))/255.
        
