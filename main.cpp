#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <stdio.h>
#include <stdlib.h>

#include <fstream>
#include <cstdlib>

#include "utils.h"
#include "objectDetect.h"

using namespace cv;
using namespace std;

const size_t inWidth = 300;
const size_t inHeight = 300;
const float WHRatio = inWidth / (float)inHeight;

int main() {

    VideoCapture capture;

    cv::VideoWriter writer;

    //writer = VideoWriter("out.avi", CV_FOURCC('X', 'V', 'I', 'D'), 30, Size(300,300));

    Config myConfig;

//    capture.open("dog3.mp4");
//
//    long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
//    cout<<"整个视频共"<<totalFrameNumber<<"帧"<<endl;
//    long FPS = capture.get(CV_CAP_PROP_FPS);
//    cout<<"FPS:"<<FPS<<endl;
//
//    FPS = FPS/2;
//
//    int i = 1;
//
//    while(totalFrameNumber - i)
//    {
//
//        Mat frame;
//        capture>>frame;
//        i++;
////        resize_image(frame, myConfig.IMAGE_MIN_DIM, myConfig.IMAGE_MAX_DIM);
//        stringstream num;
//        num << i;
//        string file_name;
//        file_name = "/home/whao/clion_ssd/cmake-build-debug/dog3/";
//        file_name = file_name + num.str() + ".jpg";
//        if(i%FPS == 0){ imwrite(file_name, frame);}
//        //cv::waitKey(3);
//        //writer.write(frame);
//    }

    string file_path = "/home/whao/clion_ssd/cmake-build-debug/dog1/";
//    string image_path = file_path + "6" + ".jpg";
//    Mat frame = imread(image_path);
//    int scale = 2;
    //resize(frame, frame, Size(scale*frame.cols,scale*frame.rows), 0, 0, INTER_LINEAR);
//    Mat out = smartSubject(NULL, 0, 0, frame);
//
//    imwrite(  "/home/whao/clion_ssd/cmake-build-debug/outs/out.jpg", out);
    int start = 14;
    for (int i = start; i <= 2184; i = i + start) {

        cout << to_string(i) << ".jpg" << endl;

        string image_path = file_path + to_string(i) + ".jpg";
        Mat frame = imread(image_path);
//        int scale = 2;
//        resize(frame, frame, Size(scale*frame.cols,scale*frame.rows), 0, 0, INTER_LINEAR);

        Mat out = smartClip(NULL, 0, 0, frame);

//        Mat out_subject = smartSubject(NULL, 0, 0, frame);

        imwrite( file_path + to_string(i) + "_c" + ".jpg", out);

//        imshow("result_SSD", image_real);
//        cv::waitKey(0);
    }
    return 0;
}