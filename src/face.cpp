// minimal2.cpp: Display the landmarks of possibly multiple faces in an image.

#include <stdio.h>
#include <stdlib.h>
#include "opencv/highgui.h"
#include "stasm_lib.h"
#include "opencv2/core/core.hpp"
#include <iostream>     // std::cout
#include <algorithm>    // std::min
#include <cmath>        // std::abs

using namespace cv;
using namespace std;


static void error(const char* s1, const char* s2)
{
    printf("Stasm version %s: %s %s\n", stasm_VERSION, s1, s2);
    exit(1);
}

double distanceCalculate(double x1, double y1, double x2, double y2) //
{
    double x = x1 - x2;
    double y = y1 - y2;
    double dist;

    dist = pow(x,2)+pow(y,2);           //calculating distance by euclidean formula
    dist = sqrt(dist);                  //sqrt is function in math.h

    return dist;
}

int main()
{
    if (!stasm_init("data", 0 /*trace*/))
        error("stasm_init failed: ", stasm_lasterr());

    static const char* path = "data/MS_480.jpg";

    cv::Mat_<unsigned char> img(cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE));

    if (!img.data)
        error("Cannot load", path);

    if (!stasm_open_image((const char*)img.data, img.cols, img.rows, path,
                          1 /*multiface*/, 10 /*minwidth*/))
        error("stasm_open_image failed: ", stasm_lasterr());

    int foundface;
    float landmarks[2 * stasm_NLANDMARKS]; // x,y coords (note the 2)

    int nfaces = 0;
    while (1)
    {
        if (!stasm_search_auto(&foundface, landmarks))
             error("stasm_search_auto failed: ", stasm_lasterr());

        if (!foundface)
            break;      // note break

        // for demonstration, convert from Stasm 77 points to XM2VTS 68 points
        stasm_convert_shape(landmarks, 77);

//        // draw the landmarks on the image as white dots
//        stasm_force_points_into_image(landmarks, img.cols, img.rows);
//        for (int i = 0; i < stasm_NLANDMARKS; i++)
//        	img(cvRound(landmarks[38*2+1]), cvRound(landmarks[38*2])) = 255;
//      		img(cvRound(landmarks[39*2+1]), cvRound(landmarks[39*2])) = 255;

        Point LPupil;
        LPupil.y = cvRound(landmarks[38*2+1]);
        LPupil.x = cvRound(landmarks[38*2]);

        Point RPupil;
        RPupil.y = cvRound(landmarks[39*2+1]);
        RPupil.x = cvRound(landmarks[39*2]);

        Point CNoseTip;
        CNoseTip.y = cvRound(landmarks[52*2+1]);
        CNoseTip.x = cvRound(landmarks[52*2]);

        Point LEyebrowInner;
        LEyebrowInner.y = cvRound(landmarks[21*2+1]);
        LEyebrowInner.x = cvRound(landmarks[21*2]);

        Point CNoseBase;
        CNoseBase.y = cvRound(landmarks[56*2+1]);
        CNoseBase.x = cvRound(landmarks[56*2]);

        Point CTipOfChin;
        CTipOfChin.y = cvRound(landmarks[6*2+1]);
        CTipOfChin.x = cvRound(landmarks[6*2]);

        // draw a line between the two eyes
        line(img, RPupil, LPupil, cvScalar(255,0,255), 1);

        // draw a line between the right eye pupil and the nose tip
        line(img, RPupil, CNoseTip, cvScalar(255,0,255), 1);

        // draw a line between the left eye pupil and the nose tip
        line(img, LPupil, CNoseTip, cvScalar(255,0,255), 1);

        // draw a line between the right eye pupil and the nose tip
        line(img, LEyebrowInner, CNoseTip, cvScalar(255,0,255), 1);

        // draw a line between the left eye pupil and the nose tip
        line(img, CNoseBase, CTipOfChin, cvScalar(255,0,255), 1);

        // roll
        double theta = atan2((double)LPupil.y - RPupil.y, LPupil.x - RPupil.x) * 180 / CV_PI;
        printf("theta = %f degrees\n", theta);

        double roll = min(2 * theta / CV_PI, 1.0); // http://www.cplusplus.com/reference/algorithm/min/
        printf("roll = %f\n", roll);

        // yaw ()
        // cálculo do dl e dr (http://answers.opencv.org/question/14188/calc-eucliadian-distance-between-two-single-point/)
        double dl = cv::norm(LPupil - CNoseTip);
        double dr = cv::norm(RPupil - CNoseTip);

        double yaw = (max(dl, dr) - min(dl, dr)) / max(dl, dr);
        printf("yaw = %f\n", yaw);

        // pitch
        double eu = cv::norm(LEyebrowInner - CNoseTip);
        double cd = cv::norm(CNoseBase - CTipOfChin);

        double pitch = (max(eu, cd) - min(eu, cd)) / max(eu, cd);
        printf("pitch = %f\n", pitch);

        // SP
        // being alpha = 0.1, beta = 0.6 and gamma = 0.3 | article page 153
        double alpha = 0.1;
        double beta = 0.6;
        double gamma = 0.3;

        double sp = alpha*(1-roll) + beta*(1-yaw) + gamma*(1-pitch);
        printf("sp = %f\n######################\n", sp);


        nfaces++;
    }

    // point coordinates according to stasm_landmarks.h



    printf("%s: %d face(s)\n", path, nfaces);
    fflush(stdout);
    cv::imwrite("minimal2.bmp", img);
    cv::imshow("stasm minimal2", img);
    cv::waitKey();

    return 0;
}
