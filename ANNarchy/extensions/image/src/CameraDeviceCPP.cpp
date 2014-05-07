#include "CameraDeviceCPP.h"

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
