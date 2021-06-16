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
		SetSetConsoleTextColor(GREEN);
		std::cout << "****************************TEST_GUASSIAN_FILTER:**************************** \n ";
		SetSetConsoleTextColor(DEFAULT);

		Time time;
		int ksize = SIZE;//kernel size
		cv::Mat image = cv::imread("../image/lena.jpg", 1);
		auto imagesize = image.size();
		DescriptionImage(image);
		cv::Mat cvdst;
		time.start(platform_cv_time);
		//LOOP_100
		cv::GaussianBlur(image, cvdst, cv::Size(ksize, ksize), SIGMA_X, SIGMA_Y);
		time.end();

		cv::Mat matCpuDst(cvdst.size(), cvdst.type());
		cv::Mat matGpuDst(cvdst.size(), cvdst.type());
		time.start(platform_cpu_time);
		//LOOP_100
		cvlib::GaussianBlur(image.data, image.size().height, image.size().width, image.channels(), ksize, SIGMA_X, SIGMA_Y, matCpuDst.data);
		time.end();

		time.start(platform_gpu_time);
		//LOOP_100
		cvlib::cuda::GaussianBlur(image.data, image.size().height, image.size().width, image.channels(), ksize, SIGMA_X, SIGMA_Y, matGpuDst.data);
		time.end();

		Check(matCpuDst, cvdst, platform_cpu);
		Check(matGpuDst, cvdst, platform_gpu);
	}
private:
};

