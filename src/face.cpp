// minimal2.cpp: Display the landmarks of possibly multiple faces in an image.

#include <stdio.h>
#include <stdlib.h>
#include "opencv/highgui.h"
#include "stasm_lib.h"
#include "opencv2/core/core.hpp"
#include <iostream>     // std::cout
#include <algorithm>    // std::min
#include <cmath>        // std::abs
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// http://stackoverflow.com/questions/15771512/compare-histograms-of-grayscale-images-in-opencv
void show_grayscale_histogram(std::string const& name, cv::Mat1b const& image)
{
    // Set histogram bins count
    int bins = 256;
    int histSize[] = {bins};

    // Set ranges for histogram bins
    float lranges[] = {0, 256};
    const float* ranges[] = {lranges};

    // create matrix for histogram
    cv::Mat hist;
    int channels[] = {0};

    // create matrix for histogram visualization
    int const hist_height = 256;
    cv::Mat3b hist_image = cv::Mat3b::zeros(hist_height, bins);

    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

    double max_val=0;
    minMaxLoc(hist, 0, &max_val);

    // visualize each bin
    for(int b = 0; b < bins; b++) {
        float const binVal = hist.at<float>(b);
        int   const height = cvRound(binVal*hist_height/max_val);
        cv::line
            ( hist_image
            , cv::Point(b, hist_height-height), cv::Point(b, hist_height)
            , cv::Scalar::all(255)
            );
    }
    //cv::imshow(name, hist_image);
}

static void error(const char* s1, const char* s2)
{
    printf("Stasm version %s: %s %s\n", stasm_VERSION, s1, s2);
    exit(1);
}

Point midpoint(double x1, double y1, double x2, double y2)
{
	double newX = (x1 + x2) / 2;
	double newY = (y1 + y2) / 2;

	return Point(newX, newY);
}


int main()
{
    if (!stasm_init("data", 0 /*trace*/))
        error("stasm_init failed: ", stasm_lasterr());

    static const char* path = "data/sofiaporto_tipopasse.png";

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

        Point LPupil = Point(cvRound(landmarks[38*2]), cvRound(landmarks[38*2+1]));
        Point RPupil = Point(cvRound(landmarks[39*2]), cvRound(landmarks[39*2+1]));
        Point CNoseTip = Point(cvRound(landmarks[52*2]), cvRound(landmarks[52*2+1]));
        Point LEyebrowInner = Point(cvRound(landmarks[21*2]), cvRound(landmarks[21*2+1]));
        Point CNoseBase = Point(cvRound(landmarks[56*2]), cvRound(landmarks[56*2+1]));
        Point CTipOfChin = Point(cvRound(landmarks[6*2]), cvRound(landmarks[6*2+1]));

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

        // SI
        // SI = 1 - F(std(mc))

        // Finding the 8 points
        Point p1 = midpoint((double)cvRound(landmarks[0*2]), (double)cvRound(landmarks[0*2+1]), (double)cvRound(landmarks[58*2]), (double)cvRound(landmarks[58*2+1]));
        Point p2 = midpoint((double)cvRound(landmarks[3*2]), (double)cvRound(landmarks[3*2+1]), (double)cvRound(landmarks[58*2]), (double)cvRound(landmarks[58*2+1]));
        Point p3 = LEyebrowInner;
        Point p4 = midpoint((double)LEyebrowInner.x, (double)LEyebrowInner.y, (double)CNoseTip.x, (double)CNoseTip.y);
        Point p5 = CNoseTip;
        Point p6 = CTipOfChin;
        Point p7 = midpoint((double)cvRound(landmarks[54*2]), (double)cvRound(landmarks[54*2+1]), (double)cvRound(landmarks[12*2]), (double)cvRound(landmarks[12*2+1]));
        Point p8 = midpoint((double)cvRound(landmarks[54*2]), (double)cvRound(landmarks[54*2+1]), (double)cvRound(landmarks[9*2]), (double)cvRound(landmarks[9*2+1]));

        // variancia dos 8 pontos através de sigmoid
        // For each such reference point, we select the corresponding region w of the image, whose size is proportional to the square containing the whole face.
        // obter os pontos de uma sub-área da imagem cujo centro de massa é cada um dos 8 pontos

        // http://stackoverflow.com/questions/12369697/access-sub-matrix-of-a-multidimensional-mat-in-opencv
        // Parameters:
        // x – x-coordinate of the top-left corner.
        // y – y-coordinate of the top-left corner (sometimes bottom-left corner).
        // width – width of the rectangle.
        // height – height of the rectangle.

        Mat subMatPt1 = img(cv::Rect(p1.x - (cvRound(img.cols*0.1)/2), p1.y - (cvRound(img.rows*0.1)/2), cvRound(img.cols*0.1), cvRound(img.rows*0.1)));
        Mat subMatPt2 = img(cv::Rect(p2.x - (cvRound(img.cols*0.1)/2), p2.y - (cvRound(img.rows*0.1)/2), cvRound(img.cols*0.1), cvRound(img.rows*0.1)));
        Mat subMatPt3 = img(cv::Rect(p3.x - (cvRound(img.cols*0.1)/2), p3.y - (cvRound(img.rows*0.1)/2), cvRound(img.cols*0.1), cvRound(img.rows*0.1)));
        Mat subMatPt4 = img(cv::Rect(p4.x - (cvRound(img.cols*0.1)/2), p4.y - (cvRound(img.rows*0.1)/2), cvRound(img.cols*0.1), cvRound(img.rows*0.1)));
        Mat subMatPt5 = img(cv::Rect(p5.x - (cvRound(img.cols*0.1)/2), p5.y - (cvRound(img.rows*0.1)/2), cvRound(img.cols*0.1), cvRound(img.rows*0.1)));
        Mat subMatPt6 = img(cv::Rect(p6.x - (cvRound(img.cols*0.1)/2), p6.y - (cvRound(img.rows*0.1)/2), cvRound(img.cols*0.1), cvRound(img.rows*0.1)));
        Mat subMatPt7 = img(cv::Rect(p7.x - (cvRound(img.cols*0.1)/2), p7.y - (cvRound(img.rows*0.1)/2), cvRound(img.cols*0.1), cvRound(img.rows*0.1)));
        Mat subMatPt8 = img(cv::Rect(p8.x - (cvRound(img.cols*0.1)/2), p8.y - (cvRound(img.rows*0.1)/2), cvRound(img.cols*0.1), cvRound(img.rows*0.1)));

        rectangle( img,
                   Point( p1.x - cvRound(img.cols*0.1/2), p1.y - (cvRound(img.rows*0.1/2))),
                   Point( p1.x + cvRound(img.cols*0.1/2), p1.y + cvRound(img.rows*0.1/2)),
                   Scalar( 0, 255, 255 ),
                   1,
                   1 );

        rectangle( img,
                   Point( p2.x - cvRound(img.cols*0.1/2), p2.y - (cvRound(img.rows*0.1/2))),
                   Point( p2.x + cvRound(img.cols*0.1/2), p2.y + cvRound(img.rows*0.1/2)),
                   Scalar( 0, 255, 255 ),
                   1,
                   1 );

        rectangle( img,
                   Point( p3.x - cvRound(img.cols*0.1/2), p3.y - (cvRound(img.rows*0.1/2))),
                   Point( p3.x + cvRound(img.cols*0.1/2), p3.y + cvRound(img.rows*0.1/2)),
                   Scalar( 0, 255, 255 ),
                   1,
                   1 );

        rectangle( img,
                   Point( p4.x - cvRound(img.cols*0.1/2), p4.y - (cvRound(img.rows*0.1/2))),
                   Point( p4.x + cvRound(img.cols*0.1/2), p4.y + cvRound(img.rows*0.1/2)),
                   Scalar( 0, 255, 255 ),
                   1,
                   1 );

        rectangle( img,
                   Point( p5.x - cvRound(img.cols*0.1/2), p5.y - (cvRound(img.rows*0.1/2))),
                   Point( p5.x + cvRound(img.cols*0.1/2), p5.y + cvRound(img.rows*0.1/2)),
                   Scalar( 0, 255, 255 ),
                   1,
                   1 );
        
        rectangle( img,
                   Point( p6.x - cvRound(img.cols*0.1/2), p6.y - (cvRound(img.rows*0.1/2))),
                   Point( p6.x + cvRound(img.cols*0.1/2), p6.y + cvRound(img.rows*0.1/2)),
                   Scalar( 0, 255, 255 ),
                   1,
                   1 );

        rectangle( img,
                   Point( p7.x - cvRound(img.cols*0.1/2), p7.y - (cvRound(img.rows*0.1/2))),
                   Point( p7.x + cvRound(img.cols*0.1/2), p7.y + cvRound(img.rows*0.1/2)),
                   Scalar( 0, 255, 255 ),
                   1,
                   1 );

        rectangle( img,
                   Point( p8.x - cvRound(img.cols*0.1/2), p8.y - (cvRound(img.rows*0.1/2))),
                   Point( p8.x + cvRound(img.cols*0.1/2), p8.y + cvRound(img.rows*0.1/2)),
                   Scalar( 0, 255, 255 ),
                   1,
                   1 );

        cv::imwrite("histograms/1.png",subMatPt1); // save
        cv::imwrite("histograms/2.png",subMatPt2); // save
        cv::imwrite("histograms/3.png",subMatPt3); // save
        cv::imwrite("histograms/4.png",subMatPt4); // save
        cv::imwrite("histograms/5.png",subMatPt5); // save
        cv::imwrite("histograms/6.png",subMatPt6); // save
        cv::imwrite("histograms/7.png",subMatPt7); // save
        cv::imwrite("histograms/8.png",subMatPt8); // save

        // histograma da área
        // ver http://docs.opencv.org/doc/tutorials/imgproc/histograms/histogram_calculation/histogram_calculation.html
        show_grayscale_histogram("w1 hist", subMatPt1);
        show_grayscale_histogram("w2 hist", subMatPt2);
        show_grayscale_histogram("w3 hist", subMatPt3);
        show_grayscale_histogram("w4 hist", subMatPt4);
        show_grayscale_histogram("w5 hist", subMatPt5);
        show_grayscale_histogram("w6 hist", subMatPt6);
        show_grayscale_histogram("w7 hist", subMatPt7);
        show_grayscale_histogram("w8 hist", subMatPt8);

        // comparar histogramas http://docs.opencv.org/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html

        nfaces++;
    }

    // Criar a matriz de origem e destino
    Mat src; // a origem é cada uma das 8 anteriores

    // separar a imagem


    printf("%s: %d face(s)\n", path, nfaces);
    fflush(stdout);
    cv::imwrite("minimal2.bmp", img);
    cv::imshow("stasm", img);
    cv::waitKey(0);

    return 0;
}
