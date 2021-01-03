#pragma once

#include "boxfilter.h"
#include "test_common_define.h"

#define SIZE 33

class TEST_BOX_FILTER
{
public:
	static void Run()
	{
		std::cout << "TEST_BOX_FILTER: \n ";

		Time time;
		int ksize = SIZE;//kernel size
		cv::Mat image = cv::imread("../image/lena.jpg", 1);
		auto imagesize = image.size();
		std::cout << "height= " << imagesize.height << "  ,width= " << imagesize.width
			<< "  ,channel= " << image.channels() << "\n";

		cv::Mat dst(image.size(), image.type());

		cv::Mat border;//�������ұ߽�����
		cv::copyMakeBorder(image, border, ksize / 2, ksize / 2, ksize / 2, ksize / 2, cv::BORDER_DEFAULT);
		
		time.start(std::string("cpu������ʱ��"));
		//LOOP_100
		cvlib::boxFilter(border.data, border.size().height, border.size().width, border.channels(), ksize,
			dst.data, dst.size().height, dst.size().width);
		time.end();

		time.start(std::string("gpu������ʱ��"));
		//LOOP_100
		cvlib::cuda::boxFilter(border.data, border.size().height, border.size().width, border.channels(), ksize,
			dst.data, dst.size().height, dst.size().width);
		time.end();

		cv::Mat cvdst;
		time.start(std::string("opencv������ʱ��"));
		//LOOP_100
		cv::boxFilter(image, cvdst, -1, cv::Size(SIZE, SIZE)/*,cv::Point(-1,-1),true*/);
		time.end();

		cv::Mat error = abs(cvdst - dst);
		cv::threshold(error, error, 2, 255, cv::THRESH_TOZERO);// ������һ�����ص����
		error.convertTo(error, CV_32F);

		double err = sum(error.mul(error))[0];
		std::cout << "���ƽ���� error��" << err << "\n";
	}
private:
};


