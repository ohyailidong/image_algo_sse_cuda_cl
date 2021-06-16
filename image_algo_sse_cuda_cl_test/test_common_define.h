#ifndef TEST_COMMON_DEFINE_H
#define TEST_COMMON_DEFINE_H
#include <iostream>
#include <string>
#include <windows.h>
#include <opencv2/world.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

const std::string platform_cpu = std::string("platform is CPU");
const std::string platform_gpu = std::string("platform is GPU");
const std::string platform_sse = std::string("platform is SSE");
const std::string platform_cl  = std::string("platform is OpenCL");

const std::string platform_cpu_time = std::string("cpu计算总时间");
const std::string platform_gpu_time = std::string("gpu计算总时间");
const std::string platform_sse_time = std::string("sse计算总时间");
const std::string platform_cl_time  = std::string("opencl计算总时间");
const std::string platform_cv_time  = std::string("opencv计算总时间");

class Time {
public:
	Time() {}
	~Time() {}
	void start(std::string des = "")
	{
		m_des = des;
		m_start = cv::getTickCount();
	}
	void end()
	{
		m_time = double(cv::getTickCount() - m_start)/cv::getTickFrequency()*1000;
		std::cout << m_des << ", time :" << m_time << "ms.\n";
	}
private:
	double m_time;
	double m_start;
	std::string m_des;
};

#define LOOP_100 for(int loop=0;loop<100;loop++)
enum ConsoleTextColor
{
	DEFAULT = 0,
	RED =1,
	GREEN =2,
	BLUE =3,
};
inline void SetSetConsoleTextColor(ConsoleTextColor index) {
	switch (index)
	{
	case DEFAULT:
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
		break;
	case RED:
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED);
		break;
	case GREEN:
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_GREEN);
		break;
	case BLUE:
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_BLUE);
		break;
	default:
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
		break;
	}
}

template<typename T>
void thres_(T* srcData, int length, int val) {
	for (int i = 0; i < length; ++i)
		srcData[i] = srcData[i] > val ? srcData[i] : 0;
}
inline void Thres(cv::Mat& err, int val) {
	int depth = err.depth();
	void* pdata = err.data;
	int length = err.rows* err.cols* err.channels();
	if (depth == CV_8U)
		thres_<uchar>((uchar*)pdata, length, val);
	else if (depth == CV_8S)
		thres_<char>((char*)pdata, length, val);
	else if (depth == CV_16U)
		thres_<unsigned short>((unsigned short*)pdata, length, val);
	else if (depth == CV_16S)
		thres_<short>((short*)pdata, length, val);
	else if (depth == CV_32S)
		thres_<int>((int*)pdata, length, val);
	else if (depth == CV_32F)
		thres_<float>((float*)pdata, length, val);
	else if (depth == CV_64F)
		thres_<double>((double*)pdata, length, val);
	else
		assert(false);
}
inline void Check(cv::Mat dst, cv::Mat cvdst, std::string des ="")
{
	cv::Mat error = abs(cvdst - dst);
	Thres(error, 2);// 最大会有两个像素的误差
	error.convertTo(error, CV_32F);
	double err =0;
	for (int i = 0; i < error.channels(); ++i) 
		err += sum(error.mul(error))[i];
	if (err > 1e-6)
		SetSetConsoleTextColor(RED);
	//else
	//	SetSetConsoleTextColor(GREEN);
	std::cout << des<<",误差平方和 error：" << err << "\n";
	SetSetConsoleTextColor(DEFAULT);
}
inline void DescriptionImage(cv::Mat& image) {
	std::cout << "height= " << image.rows << ", width= " << image.cols << ", channel= " << image.channels() << "\n";
}
#endif
