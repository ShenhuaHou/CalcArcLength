#include "EdgeDetect.h"

#include <iostream>
#include<vector>
#include<set>
#include<algorithm>
#include<string>
//#include<math>
//using namespace std;


void my_sobel(const cv::Mat_<uchar>& src, cv::Mat_<uchar>& dst, int direction)
{
	cv::Mat_<uchar> kernel;
	int radius = 0;

	// Create the kernel
	if (direction == 0)
	{
		// Sobel 3x3 X kernel
		kernel = (cv::Mat_<uchar>(3, 3) << -1, 0, +1, -2, 0, +2, -1, 0, +1);
		radius = 1;
	}
	else
	{
		// Sobel 3x3 Y kernel
		kernel = (cv::Mat_<uchar>(3, 3) << -1, -2, -1, 0, 0, 0, +1, +2, +1);
		radius = 1;
	}

	// Handle border issues
	cv::Mat_<uchar> _src;
	cv::copyMakeBorder(src, _src, radius, radius, radius, radius, cv::BORDER_REFLECT101);

	// Create output matrix
	dst.create(src.rows, src.cols);

	// Convolution loop

	// Iterate on image 
	for (int r = radius; r < _src.rows - radius; ++r)
	{
		for (int c = radius; c < _src.cols - radius; ++c)
		{
			short s = 0;

			// Iterate on kernel
			for (int i = -radius; i <= radius; ++i)
			{
				for (int j = -radius; j <= radius; ++j)
				{
					s += _src(r + i, c + j) * kernel(i + radius, j + radius);
				}
			}
			dst(r - radius, c - radius) = s;
		}
	}
}


void SobelCalc(cv::Mat& src, cv::Mat& dst)
{
	cv::Mat grad_x, grad_y;
	cv::Mat abs_grad_x, abs_grad_y;

	cv::Sobel(src, grad_x, CV_16S, 1, 0, 3, 1, 1, cv::BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	Sobel(src, grad_y, CV_16S, 0, 1, 3, 1, 1, cv::BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
}

void EnhancePic(cv::Mat& src, cv::Mat& dst)
{
	/// Create a matrix of the same type and size as src (for dst)
	dst.create(src.size(), src.type());

	double histogram[256] = { 0 };


	//statistics
	for (int x = 0; x <src.rows; x++)
	{
		for (int y = 0; y < src.cols; y++)
		{
			uchar pixel = src.at<uchar>(x, y);
			histogram[pixel]++;
		}
	}

	///calculate probability
	double p[256] = { 0 };
	for (int i = 0; i < 256; i++)
	{
		p[i] = histogram[i] / (src.rows*src.cols);
	}

	///CPF
	double cp[256] = { 0 };
	for (int x = 0; x < 256; x++)
	{
		for (int y = 0; y < x; y++)
		{
			cp[x] += p[y];
		}
	}

	///histogram transformation

	for (int x = 0; x < src.rows; x++)
	{
		for (int y = 0; y < src.cols; y++)
		{
			uchar pixel = src.at<uchar>(x, y);
			dst.at<uchar>(x, y) = (cp[pixel] * 255 + 0.5);
		}
	}
}

void LaplacianCalc(cv::Mat& src, cv::Mat& dst)
{
	cv::Mat dst_;
	Laplacian(src, dst_, CV_16S, 3, 1, 0, cv::BORDER_DEFAULT);
	convertScaleAbs(dst_, dst);
}

void PrewittCalc(cv::Mat& src, cv::Mat& dst)
{
	float prewittx[9] =
	{
		-1, 0, 1,
		-1, 0, 1,
		-1, 0, 1
	};
	float prewitty[9] =
	{
		1, 1, 1,
		0, 0, 0,
		-1, -1, -1
	};
	cv::Mat px = cv::Mat(3, 3, CV_32F, prewittx);
	cv::Mat py = cv::Mat(3, 3, CV_32F, prewitty);
	cv::Mat dstx = cv::Mat(src.size(), src.type(), src.channels());
	cv::Mat dsty = cv::Mat(src.size(), src.type(), src.channels());
	filter2D(src, dstx, src.depth(), px);
	filter2D(src, dsty, src.depth(), py);
	float tempx, tempy, temp;
	for (int i = 0; i<src.rows; i++)
	{
		for (int j = 0; j<src.cols; j++)
		{
			tempx = dstx.at<uchar>(i, j);
			tempy = dsty.at<uchar>(i, j);
			temp = sqrt(tempx*tempx + tempy*tempy);
			uchar te = static_cast<uchar>(temp);
			dst.at<uchar>(i, j) = te;
		}
	}
}
 
bool SortBySize(std::vector<cv::Point> &v1, std::vector<cv::Point> &v2)
{
	return v1.size() > v2.size();
}

void ContoursCalc(cv::Mat& src, cv::Mat& dst)
{
	cv::RNG rng(12345);
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	std::vector<std::vector<cv::Point>> RightCoutours;
	cv::findContours(src, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	/// Draw contours
	for (int i = 0; i< contours.size(); i++)
	{
		cv::Mat drawing = cv::Mat::zeros(src.size(), CV_8UC1);
		cv::Rect r0 = cv::boundingRect(cv::Mat(contours[i]));
		///100 10000
		if ((contours[i].size()>10) && ((r0.width*r0.height) > 2000) /*&& (static_cast<float>(r0.width) / r0.height)>1.0f*/)
		{
			//cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			cv::drawContours(drawing, contours, i, cv::Scalar(255), 1, 8, hierarchy, 0, cv::Point());
			cv::Mat purifyMat = cv::Mat::zeros(src.size(), CV_8UC1);
			PurifyEdge2(drawing, purifyMat);
			char windowsName[100];
			sprintf(windowsName, "edge_%d.png", i);
			cv::imwrite(windowsName, purifyMat);
		}
		dst = drawing;
	}
}


void LoGCalc(cv::Mat& src, cv::Mat& dst)
{
	dst = cv::Mat(src.size(), src.type(), src.channels());

	float ALog[9] =
	{
		0,  1, 0,
		1, -4, 1,
		0,  1, 0
	};

	cv::Mat Log = cv::Mat(3, 3, CV_32F, ALog);
	filter2D(src, dst, src.depth(), Log);
}

void ReversalCalc(cv::Mat& src, cv::Mat& dst)
{
	dst = cv::Mat(src.size(), src.type(), src.channels());
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			dst.at<uchar>(i, j) = 255 - src.at<uchar>(i, j);
		}
	}
}


void PurifyEdge2(cv::Mat& src, cv::Mat& dst)
{
	std::vector<cv::Point> edgePoint;
	dst = cv::Mat::zeros(src.size(),CV_8UC1);
	for (int j = 0; j < src.cols; j++)
	{
		bool found = false;
		for (int i = 0; i < src.rows; i++)
		{
			if ((src.at<uchar>(i, j) == 255) && (!found))
			{
				dst.at<uchar>(i, j) = 255;
				edgePoint.push_back(cv::Point(j, i));
				found = true;
			}
		}
	}
	
}

void PurifyEdge3(cv::Mat& src, cv::Mat& dst)
{
	std::vector<cv::Point> edgePoint;
	dst = cv::Mat::zeros(src.size(), CV_8UC1);


	for (int i = 0; i < src.rows; i++)
	{
		for (int j = (src.cols / 3); j >0; j--)
		{
			if ((src.at<uchar>(i, j) >100))
			{
				dst.at<uchar>(i, j) = 255;
				break;
			}
		}
	}

	for (int j = (src.cols / 3); j < (2*src.cols / 3); j++)
	{
		bool found = false;
		for (int i = src.rows - 1; i >0; i--)
		{
			if ((src.at<uchar>(i, j) == 255) && (!found))
			{
				dst.at<uchar>(i, j) = 255;
				edgePoint.push_back(cv::Point(j, i));
				found = true;
			}
		}
	}

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = (src.cols / 3); j <src.cols; j++)
		{
			if ((src.at<uchar>(i, j) >100))
			{
				dst.at<uchar>(i, j) = 255;
				break;
			}
		}
	}
}

void PurifyEdge1(cv::Mat& src, cv::Mat& dst)
{
	std::vector<cv::Point> edgePoint;
	dst = cv::Mat::zeros(src.size(), CV_8UC1);


	for (int i = 0; i < src.rows; i++)
	{
		for (int j = (2 * src.cols / 5); j >0; j--)
		{
			if ((src.at<uchar>(i, j) >100))
			{
				dst.at<uchar>(i, j) = 255;
				break;
			}
		}
	}

	for (int j = (2*src.cols / 5); j < (3 * src.cols / 5); j++)
	{
		for (int i = src.rows-1; i >0; i--)
		{
			if ((src.at<uchar>(i, j) >100))
			{
				dst.at<uchar>(i, j) = 255;
				break;
			}
		}	
	}

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = (3 * src.cols / 5); j < (src.cols-1); j++)
		{
			if ((src.at<uchar>(i, j) >100))
			{
				dst.at<uchar>(i, j) = 255;
				break;
			}
		}
	}
}

float LengthCalc(cv::Mat& src)
{
	float sumlength = 0;
	float pre_axis_x = 0;
	float pre_axis_y = 0;
	bool first_detect = true;
	for (int j = 0; j < src.cols; j++)
	{
		for (int i = src.rows - 1; i > 0; i--)
		{
			if (src.at<uchar>(i, j) >100)
			{
				if (first_detect)
				{
					pre_axis_x = (float)i;
					pre_axis_y = (float)j;
					first_detect = false;
				}
				sumlength +=std::sqrt((pre_axis_x - (float)i) * (pre_axis_x - (float)i) 
					+ (pre_axis_y - (float)j) * (pre_axis_y - (float)j));
				pre_axis_x = (float)i;
				pre_axis_y = (float)j;
				break;
			}
		}
	}
	return sumlength;
}

int DeleteUnlessPoint(cv::Mat& src, cv::Mat& mapMat, int range, float rx, float ry)
{
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = (int)(src.cols*rx); j < (int)(src.cols*ry); j++)
		{
			if (mapMat.at<uchar>(i, j) == 255)
			{
				int x1 = i - range;
				int x2 = i + range;
				int y1 = j - range;
				int y2 = j + range;

				if (x1 < 0)
					x1 = 0;
				if (x2 > src.rows)
					x2 = src.rows;
				if (y1 < 0)
					y1 = 0;
				if (y2 > src.cols)
					y2 = src.cols;

				for (int x = x1; x < x2; x++)
				{
					for (int y = y1; y < y2; y++)
					{
						src.at<uchar>(x, y) = 0;
					}
				}

			}
		}
	}
	return 0;
}