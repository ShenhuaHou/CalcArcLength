#ifndef EDGEDETECT_H_
#define EDGEDETECT_H_

#include <opencv2\opencv.hpp>
#include<vector>
void my_sobel(const cv::Mat_<uchar>& src, cv::Mat_<uchar>& dst, int direction);
void SobelCalc(cv::Mat& src,cv::Mat& dst);
void EnhancePic(cv::Mat& src, cv::Mat& dst);
void LaplacianCalc(cv::Mat& src, cv::Mat& dst);
void PrewittCalc(cv::Mat& src, cv::Mat& dst);
void ContoursCalc(cv::Mat& src,cv::Mat& dst);

void LoGCalc(cv::Mat& src, cv::Mat& dst);
void ReversalCalc(cv::Mat& src, cv::Mat& dst);

void PurifyEdge1(cv::Mat& src, cv::Mat& dst);
void PurifyEdge2(cv::Mat& src, cv::Mat& dst);
void PurifyEdge3(cv::Mat& src, cv::Mat& dst);


float LengthCalc(cv::Mat& src);

int DeleteUnlessPoint(cv::Mat& src, cv::Mat& mapMat, int range, float rx, float ry);

#endif
