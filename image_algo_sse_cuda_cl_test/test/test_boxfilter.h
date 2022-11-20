#pragma once

#include "boxfilter.h"
#include "test_common_define.h"

#define SIZE 33

class TEST_BOX_FILTER
{
public:
	static void Run()
	{
		SetSetConsoleTextColor(GREEN);
		std::cout << "****************************TEST_BOX_FILTER:**************************** \n ";
		SetSetConsoleTextColor(DEFAULT);

		Time time;
		int ksize = SIZE;//kernel size
		cv::Mat image = cv::imread("../image/lena.jpg", 1);
		auto imagesize = image.size();
		DescriptionImage(image);

		cv::Mat cvdst;
		time.start(platform_cv_time);
		//LOOP_100
		cv::boxFilter(image, cvdst, -1, cv::Size(SIZE, SIZE)/*,cv::Point(-1,-1),true*/);
		time.end();

		cv::Mat matCpuDst(cvdst.size(), cvdst.type());
		cv::Mat matGpuDst(cvdst.size(), cvdst.type());

		time.start(platform_cpu_time);
		cvlib::boxFilter(image.data, imagesize.height, imagesize.width, image.channels(), ksize, matCpuDst.data);
		time.end();

		time.start(platform_gpu_time);
		cvlib::cuda::boxFilter(image.data, image.size().height, image.size().width, image.channels(), ksize, matGpuDst.data);
		time.end();

		Check(matCpuDst, cvdst, platform_cpu);
		Check(matGpuDst, cvdst, platform_gpu);
	}
private:
};


