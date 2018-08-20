#ifndef OBJECTDETECT_H
#define OBJECTDETECT_H

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

struct object{
    int classID;
    string className;
    float confidence;
    int box[4]; //xLeftBottom yLeftBottom xRightTop yRightTop
    int rect;

};

Mat smartClip(void* buffer, int w, int h, Mat frame, string input_path = "false", string save_path = "false");
Mat smartSubject(void* buffer, int w, int h, Mat frame, string input_path = "false", string save_path = "false");
//vector<object> smartClip(void* buffer, int w, int h, Mat frame, string input_path = "false", string save_path = "false");
void smartClip_dnnLoadModel(string path_pb, string path_protxt);

#endif