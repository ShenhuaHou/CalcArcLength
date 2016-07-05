#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "EdgeDetect.h"

using namespace cv;
using namespace std;

Mat src; 
Mat src_gray;
Mat edge_;
Mat canny_output;

int thresh = 45;
int max_thresh = 255;
RNG rng(12345);

/// Function header
void thresh_callback(int, void*);

/** @function main */
int FindSecond()
{
	/// Load source image and convert it to gray
	src = imread("Arc_part.jpg",1);

	/// Convert image to gray and blur it
	cvtColor(src, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));
	imwrite("blur2.jpg", src_gray);
	/// Create Window
	char* source_window = "Source";
	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	imshow(source_window, src);

	/// Canny
	Canny(src_gray, canny_output, thresh, thresh * 2, 3);
	Mat ContoursMat;
	ContoursCalc(canny_output, ContoursMat);
	imshow("ContoursMat",ContoursMat);


	createTrackbar(" Canny thresh:", "Source", &thresh, max_thresh, thresh_callback);
	thresh_callback(0, 0);

	waitKey(0);
	return(0);
}

/** @function thresh_callback */
void thresh_callback(int, void*)
{

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	/// Detect edges using canny
	Canny(src_gray, canny_output, thresh, thresh * 2, 3);
	imwrite("canny2.jpg",canny_output);
	Mat threMat;
	threshold(canny_output, threMat, 150, 255, THRESH_BINARY);
	//imshow("threshold", threMat);
	//imwrite("threshold.jpg", threMat);

	/// Find contours
	findContours(threMat, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Draw contours
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC1);
	for (int i = 0; i< contours.size(); i++)
	{
		cv::Rect r0 = cv::boundingRect(cv::Mat(contours[i]));
		///100 10000
		if ((contours[i].size()>60) && ((r0.width*r0.height) > 8000) && (static_cast<float>(r0.width)/r0.height)>1.0f)
		{
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, Point());
			cv::Rect r0 = cv::boundingRect(cv::Mat(contours[i]));
			cv::rectangle(drawing, r0, color, 1);
		}
		
	}

	/// Show in a window
	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	imshow("Contours", drawing);
	imshow("Source", threMat);
}

int FindOneAndThree()
{
	Mat src = imread("Arc_part.jpg", IMREAD_GRAYSCALE);
	
	/// Gaussian Blur
	GaussianBlur(src, src, Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
	imshow("GaussianBlur", src);
	imwrite("GaussianBlur.jpg", src);

	/// LoG Operator
	Mat LogMat;
	LoGCalc(src, LogMat);
	imshow("Log", LogMat);
	imwrite("Log.jpg", LogMat);

	/// Enhance the Contrast
	Mat enhanceMat;
	EnhancePic(LogMat, enhanceMat);
	imshow("enchancePic", enhanceMat);
	imwrite("enchancePic.jpg", enhanceMat);

	/// Thre
	Mat threMat;
	threshold(enhanceMat, threMat, 180, 255, THRESH_BINARY);
	imshow("threshold",threMat);
	imwrite("threshold.jpg", threMat);

	//cv::Mat dst_dilate;
	//Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	//dilate(threMat, dst_dilate, element);
	//imshow("dilate.jpg", dst_dilate);

	Mat contoursMat;
	ContoursCalc(threMat, contoursMat);
	imshow("contours",contoursMat);
	cv::waitKey(0);

	return 0;
}

void ProgressFirstEdge()
{
	Mat src = imread("edge1/edge_855.jpg", IMREAD_GRAYSCALE);
	Mat dst;
	PurifyEdge1(src, dst);
	imwrite("edge1/dst.jpg", dst);
	float srclength = LengthCalc(dst);
	cout << "edge2 length: " << srclength << endl;
	cv::waitKey(0);
}

void ProgressSecondEdge()
{
	Mat src = imread("edge2/edge_111_2.jpg", IMREAD_GRAYSCALE);
	Mat dst;
	PurifyEdge2(src, dst);
	imwrite("edge2/dst.jpg", dst);

	Mat dst2 = imread("edge2/edge_111.png", IMREAD_GRAYSCALE);
	float srclength = LengthCalc(dst2);
	cout << "edge2 length: " << srclength << endl;
	cv::waitKey(0);
}


void ProgressThirdEdge()
{
	cv::Mat src1 = imread("edge3/edge_1175.jpg", IMREAD_GRAYSCALE);
	cv::Mat src2 = imread("edge3/edge_1179.jpg", IMREAD_GRAYSCALE);
	cv::Mat src3 = imread("edge3/edge_1183.jpg", IMREAD_GRAYSCALE);
	
	cv::Mat src = cv::Mat::zeros(src1.size(), src1.type());

	for (int i = 0; i < src1.rows; i++)
	{
		for (int j = 0; j < src1.cols; j++)
		{
			if ((src1.at<uchar>(i, j) > 10) || (src2.at<uchar>(i, j) > 10) || (src3.at<uchar>(i, j) > 10))
				src.at<uchar>(i, j) = 255;
		}
	}
	imshow("src", src);
	imwrite("edge3/src.jpg", src);

	cv::Mat MapMat1 = imread("edge3/map1.jpg", IMREAD_GRAYSCALE);
	cv::Mat MapMat2 = imread("edge3/map2.jpg", IMREAD_GRAYSCALE);
	Mat dst;
	DeleteUnlessPoint(src, MapMat1, 30, 0.0, 1.0);
	DeleteUnlessPoint(src, MapMat2, 33, 0.0, 0.12);
	DeleteUnlessPoint(src, MapMat2, 25, 0.15, 0.3);
	DeleteUnlessPoint(src, MapMat2, 13, 0.5, 1);
	imshow("dst", src);
	imwrite("edge3/dst.jpg", src);

	PurifyEdge3(src, dst);
	imshow("dst2", dst);
	imwrite("edge3/dst2.jpg", dst);

	float srclength = LengthCalc(src);
	cout << "edge3 length: " << srclength << endl;
	cv::waitKey(0);
}


int main(int argc, char **argv)
{
	//FindOneAndThree();

	ProgressFirstEdge();
	//ProgressSecondEdge();
	//ProgressThirdEdge();
	return 0;
}