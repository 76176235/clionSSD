#ifndef UTILS_H
#define UTILS_H

#include <opencv2/core/core.hpp>
#include "opencv2/imgproc.hpp"
#include "Config.h"
#include <iostream>

using namespace cv;
using namespace std;


struct ALL{
    Mat fin_boxes;
    Mat fin_class_ids;
    Mat fin_class_scores;
    vector<Mat> fin_masks;
};

template<typename T>
vector<T> arange(T start, T stop, T step = 1);

pair<Mat, Mat> meshgrid(Mat& x,Mat& y);

pair<float, vector<int> > resize_image(Mat& image, int min_dim, int max_dim);


pair<Mat, vector<int> > mold_image(Mat& image, Config config);

Mat generate_anchors(int scale, pair<int, int> shape,
                     int feature_stride, Config config);

Mat generate_pyramid_anchors(Config config, vector<pair<int, int> > backbone_shapes);

Mat get_anchors(int img_height, int img_width, Config config);

void norm_boxes(Mat& inMat, float img_height, float img_width);

void denorm_boxes(Mat& inMat, float img_height, float img_width);

Mat unmold_mask(Mat mask, Mat det_box, Size original_image_shape);

ALL unmold_detections(Mat detections, Mat mrcnn_mask, Size original_image_shape,
                       Size image_shape, vector<int> window);



#endif