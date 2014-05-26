import ANNarchy.core.Global as Global
from ANNarchy.generator.Generator import extra_libs 


header_template = """
#ifndef __ANNarchy_%(class_name)s_H__
#define __ANNarchy_%(class_name)s_H__

#include "Global.h"
#include "RatePopulation.h"
using namespace ANNarchy_Global;

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class CameraDeviceCPP : public VideoCapture {

public:
    
    CameraDeviceCPP (int id, int width, int height, int depth) : VideoCapture(id){
        width_ = width;
        height_ = height;
        depth_ = depth;
        
        // Vector to store the returned image
        img_ = vector<float>(width*height*depth, 0.0);
    }

    // Grab one image
    vector<float> GrabImage();


protected:
    
    // Width and height of the image, depth_ is 1 (grayscale) or 3 (RGB)
    int width_, height_, depth_;

    // Vector of floats for the returned image
    vector<float> img_;
    
};

class %(class_name)s: public RatePopulation
{
public:
    %(class_name)s(std::string name, int nbNeurons);
    
    ~%(class_name)s();
    
    void prepareNeurons();
    
    int getNeuronCount() { return nbNeurons_; }
    
    void resetToInit(){};
    
    void localMetaStep(int neur_rank){};
    
    void globalMetaStep(){};
    
    void globalOperations(){};
    
    void record();

    // Access methods for the local variable r
    std::vector<float> getR() { return this->r_; }
    void setR(std::vector<float> r) { this->r_ = r; }

    float getSingleR(int rank) { return this->r_[rank]; }
    void setSingleR(int rank, float r) { this->r_[rank] = r; }

    std::vector< std::vector< float > >getRecordedR() { return this->recorded_r_; }                    
    void startRecordR() { this->record_r_ = true; }
    void stopRecordR() { this->record_r_ = false; }
    void clearRecordedR() { this->recorded_r_.clear(); }
    
    // Camera
    void StartCamera(int id, int width, int height, int depth);
    void GrabImage();

private:

    // CameraDevice
    CameraDeviceCPP* camera_;

    // r_ : local
    bool record_r_; 
    std::vector< std::vector<float> > recorded_r_;    

};
#endif
"""

body_template = """
#include "%(class_name)s.h"
#include "Global.h"

%(class_name)s::%(class_name)s(std::string name, int nbNeurons): RatePopulation(name, nbNeurons)
{
    rank_ = 0;
    
    // Camera
    camera_ = NULL;
    
#ifdef _DEBUG
    std::cout << "%(class_name)s::%(class_name)s called (using rank " << rank_ << ")" << std::endl;
#endif


    // r_ : local
    r_ = std::vector<float> (nbNeurons_, 0.0);
    record_r_ = false; 
    recorded_r_ = std::vector< std::vector< float > >();    

    // dt : integration step
    dt_ = 1.0;


    try
    {
        Network::instance()->addPopulation(this);
    }
    catch(std::exception e)
    {
        std::cout << "Failed to attach population"<< std::endl;
        std::cout << e.what() << std::endl;
    }
}

%(class_name)s::~%(class_name)s() 
{
#ifdef _DEBUG
    std::cout << "Population0::Destructor" << std::endl;
#endif

    r_.clear();
}

void %(class_name)s::StartCamera(int id, int width, int height, int depth){

    camera_ = new CameraDeviceCPP(id, width, height, depth);
    if(!camera_->isOpened()){
        cout << "Error: could not open the camera." << endl;   
    }
}

void %(class_name)s::GrabImage(){

    if(camera_->isOpened()){
        r_ = camera_->GrabImage();   
    }
}

void %(class_name)s::prepareNeurons() 
{
    if (maxDelay_ > 1)
    {
    #ifdef _DEBUG
        std::cout << name_ << ": got delayed rates = " << maxDelay_ << std::endl;
    #endif
    
        delayedRates_.push_front(r_);
        delayedRates_.pop_back();
    }
}

void %(class_name)s::record() 
{
    if(record_r_)
        recorded_r_.push_back(r_);
}

// Camera device
// Grab one image
vector<float> CameraDeviceCPP::GrabImage(){

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
                    this->img_[(j+width_*i)*3 + 0] = float(intensity.val[2])/255.0;
                    this->img_[(j+width_*i)*3 + 1] = float(intensity.val[1])/255.0;
                    this->img_[(j+width_*i)*3 + 2] = float(intensity.val[0])/255.0;
                }
            }

        }
        
    }
    return this->img_;
}
"""

pyx_template = """
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
import numpy as np
cimport numpy as np

cdef extern from "../build/%(class_name)s.h":
    cdef cppclass %(class_name)s:
        %(class_name)s(string name, int N)

        int getNeuronCount()
        
        string getName()
        
        void resetToInit()
        
        void setMaxDelay(int)
        
        void StartCamera(int, int, int, int)
        
        void GrabImage()


        # Local r
        vector[float] getR()
        void setR(vector[float] values)
        float getSingleR(int rank)
        void setSingleR(int rank, float values)
        void startRecordR()
        void stopRecordR()
        void clearRecordedR()
        vector[vector[float]] getRecordedR()



cdef class py%(class_name)s:

    cdef %(class_name)s* cInstance

    def __cinit__(self):
        self.cInstance = new %(class_name)s('%(class_name)s', %(nb_neurons)s)

    def name(self):
        return self.cInstance.getName()

    def reset(self):
        self.cInstance.resetToInit()

    def set_max_delay(self, delay):
        self.cInstance.setMaxDelay(delay)

    property size:
        def __get__(self):
            return self.cInstance.getNeuronCount()
        def __set__(self, value):
            print "py%(class_name)s.size is a read-only attribute."
            
    # local: r
    cpdef np.ndarray _get_r(self):
        return np.array(self.cInstance.getR())
        
    cpdef _set_r(self, np.ndarray value):
        self.cInstance.setR(value)
        
    cpdef float _get_single_r(self, rank):
        return self.cInstance.getSingleR(rank)

    def _set_single_r(self, int rank, float value):
        self.cInstance.setSingleR(rank, value)

    def _start_record_r(self):
        self.cInstance.startRecordR()

    def _stop_record_r(self):
        self.cInstance.stopRecordR()

    cpdef np.ndarray _get_recorded_r(self):
        tmp = np.array(self.cInstance.getRecordedR())
        self.cInstance.clearRecordedR()
        return tmp


    # CameraDevice
    def start_camera(self, int id, int width, int height, int depth):
        self.cInstance.StartCamera(id, width, height, depth)

    def grab_image(self):
        self.cInstance.GrabImage()
"""

class VideoPopulationGenerator(object):
    """ Base class for generating C++ code from a population description. """
    def __init__(self, pop):
        self.pop = pop
        self.name = pop.class_name
        
        # Names of the files to be generated
        self.header = Global.annarchy_dir+'/generate/build/'+self.name+'.h'
        self.body = Global.annarchy_dir+'/generate/build/'+self.name+'.cpp'
        self.pyx = Global.annarchy_dir+'/generate/pyx/'+self.name+'.pyx'
        
        # Add opencv libs to the makefile
        extra_libs.append('-lopencv_core')
        extra_libs.append('-lopencv_imgproc')
        extra_libs.append('-lopencv_highgui')
        extra_libs.append('-lopencv_objdetect')
        extra_libs.append('-lopencv_video')
        
    def generate(self, verbose):

        #   generate files
        with open(self.header, mode = 'w') as w_file:
            w_file.write(header_template%{'class_name':self.name})

        with open(self.body, mode = 'w') as w_file:
            w_file.write(body_template%{'class_name':self.name})

        with open(self.pyx, mode = 'w') as w_file:
            w_file.write(pyx_template%{'class_name':self.name, 'nb_neurons':self.pop.size}) 
