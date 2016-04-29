from ANNarchy.core.Population import Population
from ANNarchy.core.Neuron import Neuron
import ANNarchy.core.Global as Global
from ANNarchy.generator.Compiler import extra_libs 

try:
    from PIL import Image
except:
    Global._warning('The Python Image Library (pillow) is not installed on your system, unable to create ImagePopulations.')
    
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
            
        if len(geometry)==3 and (geometry[2]!=3 and geometry[2]!=1):
            Global._error('The third dimension of an ImagePopulation should be either 1 (grayscale) or 3 (color).') 
                        
        if len(geometry)==3 and geometry[2]==1:
            geometry = (geometry[0], geometry[1])
            
        # Create the population     
        Population.__init__(self, geometry = geometry, name=name, neuron = Neuron(parameters="r = 0.0") )

    def set_image(self, image_name):
        """ 
        Sets an image (.png, .jpg or whatever is supported by PIL) into the firing rate of the population.
        
        If the image has a different size from the population, it will be resized.
        
        """
        try:
            im = Image.open(image_name)
        except : # image does not exist
            Global._error('The image ' + image_name + ' does not exist.')
            
        # Resize the image if needed
        (width, height) = (self.geometry[1], self.geometry[0])
        if im.size != (width, height):
            Global._warning('The image ' + image_name + ' does not have the same size '+str(im.size)+' as the population ' + str((width, height)) + '. It will be resized.')
            im = im.resize((width, height))
        # Check if only the luminance should be extracted
        if self.dimension == 2 or self.geometry[2] == 1:
            im=im.convert("L")
        # Set the rate of the population
        if not Global._network[0]['compiled']:
            self.r = (np.array(im))/255.
        else:
            self.cyInstance.set_r(np.array(im).reshape(self.size)/255.)


class VideoPopulation(ImagePopulation):
    """ 
    Specific rate-coded Population allowing to feed a webcam input into the firing rate of a population (each neuron represents one pixel).
    
    This extension requires the C++ library OpenCV >= 2.0 (apt-get/yum install opencv). ``pkg-config opencv --cflags --libs`` should not return an error.
    
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

    def _generate(self):
        "Code generation"      
        # Add corresponding libs to the Makefile
        extra_libs.append('`pkg-config opencv --cflags --libs`')

        # Include opencv
        self._specific_template['include_additional'] = """#include <opencv2/opencv.hpp>
using namespace cv;
"""
        # Class for the camera device
        self._specific_template['struct_additional'] = """
// VideoPopulation
class CameraDeviceCPP : public cv::VideoCapture {
public:
    CameraDeviceCPP (int id, int width, int height, int depth) : cv::VideoCapture(id){
        width_ = width;
        height_ = height;
        depth_ = depth;      
        img_ = vector<double>(width*height*depth, 0.0);
    }
    vector<double> GrabImage(){
        if(isOpened()){
            // Read a new frame from the video
            Mat frame;
            read(frame); 
            // Resize the image
            Mat resized_frame;
            resize(frame, resized_frame, Size(width_, height_) );
            // If depth=1, only luminance
            if(depth_==1){
                // Convert to luminance
                cvtColor(resized_frame, resized_frame, CV_BGR2GRAY);
                for(int i = 0; i < resized_frame.rows; i++){
                    for(int j = 0; j < resized_frame.cols; j++){
                        this->img_[j+width_*i] = float(resized_frame.at<uchar>(i, j))/255.0;
                    }
                }
            }
            else{ //BGR
                for(int i = 0; i < resized_frame.rows; i++){
                    for(int j = 0; j < resized_frame.cols; j++){
                        Vec3b intensity = resized_frame.at<Vec3b>(i, j);
                        this->img_[(j+width_*i)*3 + 0] = double(intensity.val[2])/255.0;
                        this->img_[(j+width_*i)*3 + 1] = double(intensity.val[1])/255.0;
                        this->img_[(j+width_*i)*3 + 2] = double(intensity.val[0])/255.0;
                    }
                }
            }            
        }
        return this->img_;
    };

protected:
    // Width and height of the image, depth_ is 1 (grayscale) or 3 (RGB)
    int width_, height_, depth_;
    // Vector of floats for the returned image
    vector<double> img_;
};
"""

        self._specific_template['declare_additional'] = """
    // Camera
    CameraDeviceCPP* camera_;
    void StartCamera(int id, int width, int height, int depth){
        camera_ = new CameraDeviceCPP(id, width, height, depth);
        if(!camera_->isOpened()){
            std::cout << "Error: could not open the camera." << std::endl;   
        }
    };
    void GrabImage(){
        if(camera_->isOpened()){
            r = camera_->GrabImage();   
        }
    };
"""

        self._specific_template['update_variables'] = ""

        self._specific_template['export_additional'] = """
        void StartCamera(int id, int width, int height, int depth)
        void GrabImage()
""" 

        self._specific_template['wrapper_access_additional'] = """
    # CameraDevice
    def start_camera(self, int id, int width, int height, int depth):
        pop%(id)s.StartCamera(id, width, height, depth)

    def grab_image(self):
        pop%(id)s.GrabImage()
""" % {'id': self.id}


            
    def start_camera(self, camera_port=0):
        """
        Starts the webcam with the corresponding device (default = 0).
        
        On linux, the camera port corresponds to the number in /dev/video0, /dev/video1, etc.
        """

        self.cyInstance.start_camera(camera_port, self.geometry[1], self.geometry[0], 3 if self.dimension==3 else 1)
        
    def grab_image(self):
        """
        Grabs one image from the camera and feeds it into the population.
        
        The camera must be first started with:
        
        >>> pop.start_camera(0)
        """
        self.cyInstance.grab_image()

        
