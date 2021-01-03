#pragma once


#include "histogram.h"
#include "test_common_define.h"

class TEST_HISTOGRAM
{
public:
	static void Run()
	{
		std::cout << "TEST_HISTOGRAM: \n ";

		Time time;
		cv::Mat image = cv::imread("../image/lena.jpg", 0);
		auto imagesize = image.size();
		std::cout << "height= " << imagesize.height << "  ,width= " << imagesize.width
			<< "  ,channel= " << image.channels() << "\n";

		cv::Mat dst(image.size(), image.type());


		time.start(std::string("cpu������ʱ��"));
		//LOOP_100
		cvlib::equalizeHist(image.data, dst.data, imagesize.height, imagesize.width);
		time.end();

		time.start(std::string("gpu������ʱ��"));
		//LOOP_100
		cvlib::cuda::equalizeHist(image.data, dst.data, imagesize.height, imagesize.width);
		time.end();

		cv::Mat cvdst;
		time.start(std::string("opencv������ʱ��"));
		//LOOP_100
		cv::equalizeHist(image, cvdst);
		time.end();

		cv::Mat error = abs(cvdst - dst);
		cv::threshold(error, error, 2, 255, cv::THRESH_TOZERO);// ������һ�����ص����
		error.convertTo(error, CV_32F);
		double err = sum(error.mul(error))[0];
		std::cout << "���ƽ���� error��" << err << "\n";
	}
private:
};

