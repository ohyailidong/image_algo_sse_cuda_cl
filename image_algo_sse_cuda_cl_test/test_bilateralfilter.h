#pragma once

#include "test_common_define.h"
#include "bilateralfilter.h"
#include "copymakeborder.h"
#define SIZE 33
#define SIGMA_COLOR 25
#define SIGMA_SPACE 12.5

class TEST_BILATERAL_FILTER
{
public:
	static void Run()
	{
		std::cout << "TEST_BILATERAL_FILTER: \n ";

		Time time;
		int ksize = SIZE;//kernel size
		cv::Mat image = cv::imread("../image/bilateral.png", 1);
		auto imagesize = image.size();
		std::cout << "height= " << imagesize.height << "  ,width= " << imagesize.width
			<< "  ,channel= " << image.channels() << "\n";

		cv::Mat dst(image.size(), image.type());
		time.start(std::string("cpu计算总时间"));
		//LOOP_100
		cvlib::bilateralFilter(image.data, image.rows, image.cols, image.channels(), ksize, 
			SIGMA_COLOR, SIGMA_SPACE,dst.data);
		time.end();

		time.start(std::string("gpu计算总时间"));
		//LOOP_100
		cvlib::cuda::bilateralFilter(image.data, image.rows, image.cols, image.channels(), ksize, 
			SIGMA_COLOR, SIGMA_SPACE,dst.data);
		time.end();

		cv::Mat cvdst;
		time.start(std::string("opencv计算总时间"));
		//LOOP_100
		cv::bilateralFilter(image, cvdst, ksize, SIGMA_COLOR, SIGMA_SPACE/*,cv::Point(-1,-1),true*/);
		time.end();
		Check(cvdst, dst);
	}
private:
};


