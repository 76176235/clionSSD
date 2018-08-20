#include "objectDetect.h"

using namespace cv;
using namespace std;
dnn::Net net;

const size_t inWidth = 300;
const size_t inHeight = 300;
const float WHRatio = inWidth / (float)inHeight;

String modelFile = "./frozen_inference_graph.pb";
String prototextFile = "./ssd_mobilenet_v1_coco.pbtxt";

const char* classNamesSSD[] = {
        "BG0", "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "BG1", "stop sign", "parking meter", "bench", "bird",
        "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
        "zebra", "giraffe", "BG2", "backpack", "umbrella", "BG3", "BG4", "handbag", "tie",
        "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "BG5", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
        "donut", "cake", "chair", "couch", "potted plant", "bed", "BG6",
        "dining table", "BG7", "BG8", "toilet", "BG9", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster",
        "sink", "refrigerator", "BG10", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"
};

bool comp(const object &a, const object &b){
    return a.rect > b.rect;
}

void smartClip_dnnLoadModel(string path_pb, string path_protxt)
{
//    String path_pb = "/storage/emulated/0/2/ssd_mobilenet/frozen_inference_graph.pb";
//    String path_protxt = "/storage/emulated/0/2/ssd_mobilenet/ssd_mobilenet_v1_coco.pbtxt";

    net = dnn::readNetFromTensorflow(path_pb, path_protxt);
}

//vector<object> smartClip(void* buffer, int w, int h, Mat frame, string input_path, string save_path)
Mat smartClip(void* buffer, int w, int h, Mat frame, string input_path, string save_path)
{
    smartClip_dnnLoadModel(modelFile, prototextFile);

//    Mat frame;
    w = frame.cols;
    h = frame.rows;
//    if(input_path != "false")
//    {
//
////        frame = imread(input_path);
//        w = frame.cols;
//        h = frame.rows;
//    }
//    else
//    {
//        Mat temp(h, w, CV_8UC3,(uchar *) buffer);
//        frame = temp;
//    }

    if(save_path != "false")
    {
        imwrite(save_path, frame);
    }
    Mat frame_out = frame.clone();
    pair<float, vector<int> > scale_window = resize_image(frame, 300, 300);
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1. / 255, Size(300, 300));

    net.setInput(blob);
    Mat output = net.forward();

    Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

    vector<object> fin_out;
    //vector<object>().swap(fin_out);

    int img_height = frame.rows;
    int img_width = frame.cols;
    Mat mat_window = Mat(1, 4, CV_32SC1, scale_window.second.data());
    mat_window.convertTo(mat_window, CV_32FC1);
    norm_boxes(mat_window, img_height, img_width);

    float wy1 = mat_window.at<float>(0, 0), wx1 = mat_window.at<float>(0, 1),
            wy2 = mat_window.at<float>(0, 2), wx2 = mat_window.at<float>(0, 3);

    float shift[4] = {wy1, wx1, wy1, wx1};
    Mat mat_shift = Mat(1, 4, CV_32FC1, &shift);
    float scale[4] = {wy2 - wy1, wx2 - wx1, wy2 - wy1, wx2 - wx1};
    Mat mat_scale = Mat(1, 4, CV_32FC1, &scale);

    for (int i = 0; i < detectionMat.rows; i++) {
        bool flagClassTH = false;

        if(detectionMat.at<float>(i, 1) == 1){          //person
            if(detectionMat.at<float>(i, 2) >= 0.39) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 17){    //cat
            if(detectionMat.at<float>(i, 2) >= 0.02) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 6){    //bus
            if(detectionMat.at<float>(i, 2) >= 0.8) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 18){    //dog
            if(detectionMat.at<float>(i, 2) >= 0.2) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 3){    //car
            if(detectionMat.at<float>(i, 2) >= 0.4) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 24){   //zebra
            if(detectionMat.at<float>(i, 2) >= 0.2) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 43){   //tennis racket
            if(detectionMat.at<float>(i, 2) >= 0.18) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 25){   //giraffe
            if(detectionMat.at<float>(i, 2) >= 0.046) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 60){   //donut
            if(detectionMat.at<float>(i, 2) >= 0.1) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 20){   //sheep    0.39
            if(detectionMat.at<float>(i, 2) >= 0.9) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 5){   //airplane
            if(detectionMat.at<float>(i, 2) >= 0.5) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 2){   //bicycle
            if(detectionMat.at<float>(i, 2) >= 0.1) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 4){   //motorcycle
            if(detectionMat.at<float>(i, 2) >= 0.1) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 21){   //cow       0.4
            if(detectionMat.at<float>(i, 2) >= 0.9) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 9){   //boat
            if(detectionMat.at<float>(i, 2) >= 0.14) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 31){   //handbag
            if(detectionMat.at<float>(i, 2) >= 0.2) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 10){   //traffic light
            if(detectionMat.at<float>(i, 2) >= 0.29) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 15){   //bench
            if(detectionMat.at<float>(i, 2) >= 0.35) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 65){   //bed
            if(detectionMat.at<float>(i, 2) >= 0.5) {
                flagClassTH = true;
            }
        }
        else
        {
            if(detectionMat.at<float>(i, 2) >= 0.45) {
                flagClassTH = true;
            }
        }
        if (flagClassTH) {

            float boxes[4] = {detectionMat.at<float>(i, 4), detectionMat.at<float>(i, 3),
                              detectionMat.at<float>(i, 6), detectionMat.at<float>(i, 5)};
            Mat det_boxes = Mat(1, 4, CV_32FC1, &boxes);

            det_boxes = (det_boxes - mat_shift) / mat_scale;
            denorm_boxes(det_boxes, h, w);

            bool flag = false;

            for(int j = 0; j<fin_out.size(); j++)
            {
                if((fin_out[j].box[0] == static_cast<int>(det_boxes.at<int>(0, 0)))&&\
                (fin_out[j].box[1] == static_cast<int>(det_boxes.at<int>(0, 1))&&\
                (fin_out[j].box[2] == static_cast<int>(det_boxes.at<int>(0, 2)))&&\
                (fin_out[j].box[3] == static_cast<int>(det_boxes.at<int>(0, 3)))))
                {
                    if(fin_out[j].confidence > detectionMat.at<float>(i, 2))
                    {
                        flag = true;
                        break;
                    }
                    else
                    {
                        fin_out[j].classID = static_cast<int>(detectionMat.at<float>(i, 1));
                        fin_out[j].className = string(classNamesSSD[fin_out[j].classID]);
                        fin_out[j].confidence = detectionMat.at<float>(i, 2);
                        flag = true;
                        break;
                    }
                }
            }
            if(flag == false)
            {
                object Object;
                Object.classID = static_cast<int>(detectionMat.at<float>(i, 1));
                Object.className = string(classNamesSSD[Object.classID]);
                Object.confidence = detectionMat.at<float>(i, 2);

                Object.box[0] = static_cast<int>(det_boxes.at<int>(0, 0));
                Object.box[1] = static_cast<int>(det_boxes.at<int>(0, 1));
                Object.box[2] = static_cast<int>(det_boxes.at<int>(0, 2));
                Object.box[3] = static_cast<int>(det_boxes.at<int>(0, 3));

                fin_out.push_back(Object);
            }

        }
    }

//    return fin_out;
    vector<object> fin;
    fin = fin_out;

    float confidenceThreshold = 1 * 0.01;
    for (int i = 0; i < fin.size(); i++)
    {
        float confidence = fin[i].confidence;
        if (confidence > confidenceThreshold){

            int objectClass = fin[i].classID;

            int xLeftBottom = fin[i].box[1];
            int yLeftBottom = fin[i].box[0];
            int xRightTop = fin[i].box[3];
            int yRightTop = fin[i].box[2];

            ostringstream ss;
            ss << confidence;
            String conf(ss.str());

            Rect object((int)xLeftBottom, (int)yLeftBottom,
                        (int)(xRightTop - xLeftBottom),
                        (int)(yRightTop - yLeftBottom));

            rectangle(frame_out, object, Scalar(0, 255, 0),0.5);
            String label = String(classNamesSSD[objectClass]) + ": " + conf;
            int baseLine = 0;
            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            rectangle(frame_out, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),
                                  Size(labelSize.width, labelSize.height + baseLine)),
                      Scalar(0, 255, 0), CV_FILLED);
            putText(frame_out, label, Point(xLeftBottom, yLeftBottom),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
        }
    }

    return frame_out;

}


Mat smartSubject(void *buffer, int w, int h, Mat frame_src, string input_path, string save_path)
{
//    Mat frame_src;
    smartClip_dnnLoadModel(modelFile, prototextFile);
    w = frame_src.cols;
    h = frame_src.rows;
//    if(input_path != "false")
//    {
//        frame_src = imread(input_path);
//        w = frame_src.cols;
//        h = frame_src.rows;
//    }
//    else
//    {
//        Mat temp(h, w, CV_8UC3,(uchar *) buffer);
//        frame_src = temp;
//    }

    if(save_path != "false")
    {
        imwrite(save_path, frame_src);
    }
    Mat frame = frame_src.clone();
    pair<float, vector<int> > scale_window = resize_image(frame, 300, 300);
//    imshow("frame",frame);
//    waitKey(0);
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1. / 255, Size(300, 300));

    net.setInput(blob);
    Mat output = net.forward();

    Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

    vector<object> fin_out;
    //vector<object>().swap(fin_out);

    int img_height = frame.rows;
    int img_width = frame.cols;
    Mat mat_window = Mat(1, 4, CV_32SC1, scale_window.second.data());
    mat_window.convertTo(mat_window, CV_32FC1);
    norm_boxes(mat_window, img_height, img_width);

    float wy1 = mat_window.at<float>(0, 0), wx1 = mat_window.at<float>(0, 1),
            wy2 = mat_window.at<float>(0, 2), wx2 = mat_window.at<float>(0, 3);

    float shift[4] = {wy1, wx1, wy1, wx1};
    Mat mat_shift = Mat(1, 4, CV_32FC1, &shift);
    float scale[4] = {wy2 - wy1, wx2 - wx1, wy2 - wy1, wx2 - wx1};
    Mat mat_scale = Mat(1, 4, CV_32FC1, &scale);

//    cout << "detectionMat" << detectionMat ;

    for (int i = 0; i < detectionMat.rows; i++) {
        bool flagClassTH = false;

        if(detectionMat.at<float>(i, 1) == 1){          //person
            if(detectionMat.at<float>(i, 2) >= 0.39) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 17){    //cat
            if(detectionMat.at<float>(i, 2) >= 0.02) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 6){    //bus
            if(detectionMat.at<float>(i, 2) >= 0.8) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 18){    //dog 0.2
            if(detectionMat.at<float>(i, 2) >= 0.1) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 3){    //car
            if(detectionMat.at<float>(i, 2) >= 0.4) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 24){   //zebra
            if(detectionMat.at<float>(i, 2) >= 0.2) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 43){   //tennis racket
            if(detectionMat.at<float>(i, 2) >= 0.18) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 25){   //giraffe
            if(detectionMat.at<float>(i, 2) >= 0.046) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 60){   //donut
            if(detectionMat.at<float>(i, 2) >= 0.1) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 20){   //sheep
            if(detectionMat.at<float>(i, 2) >= 0.39) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 5){   //airplane
            if(detectionMat.at<float>(i, 2) >= 0.5) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 2){   //bicycle
            if(detectionMat.at<float>(i, 2) >= 0.1) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 4){   //motorcycle
            if(detectionMat.at<float>(i, 2) >= 0.1) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 21){   //cow
            if(detectionMat.at<float>(i, 2) >= 0.4) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 9){   //boat
            if(detectionMat.at<float>(i, 2) >= 0.14) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 31){   //handbag
            if(detectionMat.at<float>(i, 2) >= 0.2) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 10){   //traffic light
            if(detectionMat.at<float>(i, 2) >= 0.29) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 15){   //bench
            if(detectionMat.at<float>(i, 2) >= 0.35) {
                flagClassTH = true;
            }
        }
        else if(detectionMat.at<float>(i, 1) == 65){   //bed
            if(detectionMat.at<float>(i, 2) >= 0.5) {
                flagClassTH = true;
            }
        }
        else
        {
            if(detectionMat.at<float>(i, 2) >= 0.45) {
                flagClassTH = true;
            }
        }
        if (flagClassTH) {

            float boxes[4] = {detectionMat.at<float>(i, 4), detectionMat.at<float>(i, 3),
                              detectionMat.at<float>(i, 6), detectionMat.at<float>(i, 5)};
            Mat det_boxes = Mat(1, 4, CV_32FC1, &boxes);

            det_boxes = (det_boxes - mat_shift) / mat_scale;
            denorm_boxes(det_boxes, h, w);

            bool flag = false;

            for(int j = 0; j<fin_out.size(); j++)
            {

                if((fin_out[j].box[0] == static_cast<int>(det_boxes.at<int>(0, 0)))&&\
            (fin_out[j].box[1] == static_cast<int>(det_boxes.at<int>(0, 1))&&\
            (fin_out[j].box[2] == static_cast<int>(det_boxes.at<int>(0, 2)))&&\
            (fin_out[j].box[3] == static_cast<int>(det_boxes.at<int>(0, 3)))))
                {
                    if(fin_out[j].confidence > detectionMat.at<float>(i, 2))
                    {
                        flag = true;
                        break;
                    }
                    else
                    {
                        fin_out[j].classID = static_cast<int>(detectionMat.at<float>(i, 1));
                        fin_out[j].className = string(classNamesSSD[fin_out[j].classID]);
                        fin_out[j].confidence = detectionMat.at<float>(i, 2);
                        flag = true;
                        break;
                    }
                }
            }
            if(flag == false)
            {
                object Object;
                Object.classID = static_cast<int>(detectionMat.at<float>(i, 1));
                Object.className = string(classNamesSSD[Object.classID]);
                Object.confidence = detectionMat.at<float>(i, 2);

                Object.box[0] = static_cast<int>(det_boxes.at<int>(0, 0));
                Object.box[1] = static_cast<int>(det_boxes.at<int>(0, 1));
                Object.box[2] = static_cast<int>(det_boxes.at<int>(0, 2));
                Object.box[3] = static_cast<int>(det_boxes.at<int>(0, 3));

                Object.rect =((Object.box[2] - Object.box[0])*(Object.box[3] - Object.box[1]));

                fin_out.push_back(Object);
            }

        }
    }

    sort(fin_out.begin(),fin_out.end(),comp);

//    return fin_out[1].box;

    int totalArea = w*h;
    sort(fin_out.begin(),fin_out.end(),comp);

    vector<int> fin_rect;
    fin_rect.push_back(0);
    fin_rect.push_back(0);
    fin_rect.push_back(0);
    fin_rect.push_back(0);

    vector<double> fin_rect_all;
    int category90[90] = {0};

    if (fin_out.size() >= 1)
    {
//        cout << " fin_out[0] :" << 1.0*fin_out[0].rect/totalArea << endl;
        if(1.0*fin_out[0].rect/totalArea > 0.036)
        {
            for(int i = 0;i<4;i++)
            {
                fin_rect[i] = fin_out[0].box[i];

            }
            category90[fin_out[0].classID] = category90[fin_out[0].classID] + 1;
            for(int j = 1; j<fin_out.size(); j++)
            {
//                cout << "totalArea:" << 1.0*fin_out[j].rect /totalArea << endl;
//                cout << "-1 :" <<  1.0*fin_out[j].rect /fin_out[j-1].rect << endl;

                if ((fin_out[j].rect > totalArea*0.0019) && (fin_out[j].rect > fin_out[j-1].rect*0.13))
                {
                    if (fin_rect[0] > fin_out[j].box[0])
                    {
                        fin_rect[0] = fin_out[j].box[0];
                    }
                    if (fin_rect[1] > fin_out[j].box[1])
                    {
                        fin_rect[1] = fin_out[j].box[1];
                    }
                    if (fin_rect[2] < fin_out[j].box[2])
                    {
                        fin_rect[2] = fin_out[j].box[2];
                    }
                    if (fin_rect[3] < fin_out[j].box[3])
                    {
                        fin_rect[3] = fin_out[j].box[3];
                    }

                    category90[fin_out[j].classID] = category90[fin_out[j].classID] + 1;

                }
                else
                {
                    break;
                }
            }

            if(fin_rect[0] < 0)
            {
                fin_rect[0] = 0;
            }
            if(fin_rect[1] < 0)
            {
                fin_rect[1] = 0;
            }
            if(fin_rect[2] > h)
            {
                fin_rect[2] = h;
            }
            if(fin_rect[3] > w)
            {
                fin_rect[3] = w;
            }

            cout <<"totalArea scale :" + to_string(1.0*((fin_rect[2] - fin_rect[0])*(fin_rect[3] - fin_rect[1]))/totalArea) << endl;

            if(1.0*((fin_rect[2] - fin_rect[0])*(fin_rect[3] - fin_rect[1]))/totalArea < 0.14)
            {
                fin_rect_all.push_back(-1);
                fin_rect_all.push_back(1);
                fin_rect_all.push_back(2);
                fin_rect_all.push_back(2);
                if(fin_out[0].confidence > 0.6)
                {
                    fin_rect_all.push_back(fin_out[0].classID);
                }
            }
            else
            {
                fin_rect_all.push_back((double)(fin_rect[1]) / (double)(w) * 2 - 1);
                fin_rect_all.push_back((double)(fin_rect[2]) / (double)(h) * 2 - 1);
                fin_rect_all.push_back((double)(fin_rect[3] - fin_rect[1]) / (double)(w) * 2);
                fin_rect_all.push_back((double)(fin_rect[2] - fin_rect[0]) / (double)(h) * 2);

                for(int i=0;i<90;i++)
                {
                    if(category90[i] > 0)
                    {
                        fin_rect_all.push_back(i);
                    }
                }
            }
        }
        else
        {
            fin_rect_all.push_back(-1);
            fin_rect_all.push_back(1);
            fin_rect_all.push_back(2);
            fin_rect_all.push_back(2);
            if(fin_out[0].confidence > 0.6)
            {
                fin_rect_all.push_back(fin_out[0].classID);
            }
        }

    }
    else
    {
        fin_rect_all.push_back(-1);
        fin_rect_all.push_back(1);
        fin_rect_all.push_back(2);
        fin_rect_all.push_back(2);
        if(fin_out[0].confidence > 0.6)
        {
            fin_rect_all.push_back(fin_out[0].classID);
        }

    }

    cout << "result :" << endl;
    for ( int i = 4;i < fin_rect_all.size();i++)
    {
        cout << classNamesSSD[int(fin_rect_all[i])] << endl;
    }

//    vector<double> fin_rect_all;

//    if(1.0*((fin_rect[2] - fin_rect[0])*(fin_rect[3] - fin_rect[1]))/totalArea < 0.14)
//    if(0)
//    {
//
//        fin_rect_all.push_back(-1);
//        fin_rect_all.push_back(1);
//        fin_rect_all.push_back(2);
//        fin_rect_all.push_back(2);
//
//    }
//    else
//    {
//
//        if(fin_rect[0] < 0)
//        {
//            fin_rect[0] = 0;
//        }
//        if(fin_rect[1] < 0)
//        {
//            fin_rect[1] = 0;
//        }
//        if(fin_rect[2] > h)
//        {
//            fin_rect[2] = h;
//        }
//        if(fin_rect[3] > w)
//        {
//            fin_rect[3] = w;
//        }
//
//        fin_rect_all.push_back((double)(fin_rect[1]) / (double)(w) * 2 - 1);
//        fin_rect_all.push_back((double)(fin_rect[2]) / (double)(h) * 2 - 1);
//        fin_rect_all.push_back((double)(fin_rect[3] - fin_rect[1]) / (double)(w) * 2);
//        fin_rect_all.push_back((double)(fin_rect[2] - fin_rect[0]) / (double)(h) * 2);
//
//    }


    vector<double > fin;
    fin = fin_rect_all;

    int xLeftBottom = frame_src.cols * (fin[0] + 1) / 2;
    int yLeftBottom = frame_src.rows * (fin[1] - fin[3] + 1) / 2;
    int xRightTop = xLeftBottom + frame_src.cols * fin[2]/2;
    int yRightTop = yLeftBottom + frame_src.rows * fin[3]/2;


    Rect object((int)xLeftBottom, (int)yLeftBottom,
                (int)(xRightTop - xLeftBottom),
                (int)(yRightTop - yLeftBottom));

    rectangle(frame_src, object, Scalar(0, 255, 0),0.5);


    return frame_src;

}
