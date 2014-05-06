#include <vector>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

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

