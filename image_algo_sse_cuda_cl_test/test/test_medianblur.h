#pragma once

#include "medianblur.h"
#include "test_common_define.h"
#define SIZE 3

class TEST_MEDIANBLUR_FILTER
{
public:
	static void Run()
	{
		SetSetConsoleTextColor(GREEN);
		std::cout << "****************************TEST_MEDIANBLUR_FILTER:**************************** \n ";
		SetSetConsoleTextColor(DEFAULT);

		Time time;
		int ksize = SIZE;//kernel size
		cv::Mat image = cv::imread("../image/lena.jpg", 1);
		auto imagesize = image.size();
		std::cout << "height= " << imagesize.height << "  ,width= " << imagesize.width
			<< "  ,channel= " << image.channels() << "\n";

		cv::Mat dst(image.size(), image.type());

		cv::Mat border;//左右边界扩充，上下不扩充
		cv::copyMakeBorder(image, border, 0, 0, ksize / 2, ksize / 2, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
		
		time.start(std::string("cpu计算总时间"));
		//LOOP_100
		cvlib::medianBlur(border.data, border.size().height, border.size().width, border.channels(), ksize,
			dst.data, dst.size().height, dst.size().width);
		time.end();

		time.start(std::string("gpu计算总时间"));
		//LOOP_100
		cvlib::cuda::medianBlur(border.data, border.size().height, border.size().width, border.channels(), ksize,
			dst.data, dst.size().height, dst.size().width);
		time.end();

		cv::Mat cvdst;
		time.start(std::string("opencv计算总时间"));
		//LOOP_100
		cv::medianBlur(image, cvdst, ksize);
		time.end();

		cv::Mat error = abs(cvdst - dst);
		cv::threshold(error, error, 2, 255, cv::THRESH_TOZERO);// 最大会有一个像素的误差
		error.convertTo(error, CV_32F);
		double err = sum(error.mul(error))[0];
		std::cout << "误差平方和 error：" << err << "\n";
	}
private:
};


