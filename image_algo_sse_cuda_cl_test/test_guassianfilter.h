#include "guassianfilter.h"
#include "test_common_define.h"

#define SIZE 7
#define SIGMA_X 2.5
#define SIGMA_Y 1.5

class TEST_GUASSIAN_FILTER
{
public:
	static void Run()
	{
		Time time;
		int ksize = SIZE;//kernel size
		cv::Mat image = cv::imread("../image/lena.jpg", 1);
		cv::Mat dst(image.size(), image.type());
		auto imagesize = image.size();
		std::cout << "height= " << imagesize.height << "  ,width= " << imagesize.width << "\n";

		time.start(std::string("cpu计算总时间"));
		//LOOP_100
		cvlib::GaussianBlur(image.data, image.size().height, image.size().width, image.channels(), ksize, SIGMA_X, SIGMA_Y, dst.data);
		time.end();

		time.start(std::string("gpu计算总时间"));
		//LOOP_100
		cvlib::cuda::GaussianBlur(image.data, image.size().height, image.size().width, image.channels(), ksize, SIGMA_X, SIGMA_Y, dst.data);
		time.end();

		cv::Mat cvdst;
		time.start(std::string("opencv计算总时间"));
		//LOOP_100
		cv::GaussianBlur(image, cvdst, cv::Size(ksize, ksize), SIGMA_X, SIGMA_Y);
		time.end();

		cv::Mat error = abs(cvdst - dst);
		cv::threshold(error, error, 2, 255, cv::THRESH_TOZERO);// 最大会有一个像素的误差
		error.convertTo(error, CV_32F);

		double minval, maxval;
		if (error.channels() == 1)
		{
			cv::Point minid, maxid;
			cv::minMaxLoc(error, &minval, &maxval, &minid, &maxid);
		}

		double err = sum(error.mul(error))[0];
		std::cout << "误差平方和 error：" << err << "\n";
	}
private:
};

