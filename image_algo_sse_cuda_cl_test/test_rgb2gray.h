#pragma once

#include "rgb2gray.h"


//void main()
//{
//	int*initcuda;
//	cudaMalloc((void**)&initcuda, 1);//每个线程首次使用cuda时，进行初始化一次，可以节约后续开辟内存空间时间
//	cudaDeviceProp prop;
//	cudaGetDeviceProperties(&prop, 0);
//
//	Time time;
//	cv::Mat image = cv::imread("lena.jpg", 1);
//	cv::Mat image0 = cv::imread("lena.jpg", 0);
//	cv::Mat dst(image.size(), CV_8UC1);
//	auto imagesize = image.size();
//	std::cout << "height= " << imagesize.height << "  ,width= " << imagesize.width << "\n";
//	time.start(std::string("RGB2GRAY 计算总时间"));
//	cvlib::cuda::RGB2GRAY(image.data, image.size().height, image.size().width, dst.data);
//	time.end();
//	time.start(std::string("RGB2GRAY_SCALE 计算总时间"));
//	cvlib::RGB2GRAY_SCALE(image.data, image.size().height, image.size().width, dst.data);
//	time.end();
//	time.start(std::string("RGB2GRAY_TBB 计算总时间"));
//	cvlib::RGB2GRAY_TBB(image.data, image.size().height, image.size().width, dst.data);
//	time.end();
//	time.start(std::string("RGB2GRAY_SIMD 计算总时间"));
//	cvlib::RGB2GRAY_SIMD(image.data, image.size().height, image.size().width, dst.data);
//	time.end();
//
//	cv::Mat cvdst;
//	time.start(std::string("opencv计算总时间"));
//	//LOOP_100
//	cv::cvtColor(image, cvdst, cv::COLOR_RGB2GRAY);
//	time.end();
//	cv::Mat error = abs(image0 - dst);
//	cv::threshold(error, error, 2, 255, cv::THRESH_TOZERO);// 最大会有一个像素的误差
//	error.convertTo(error, CV_32F);
//
//	double minval, maxval;
//	if (error.channels() == 1)
//	{
//		cv::Point minid, maxid;
//		cv::minMaxLoc(error, &minval, &maxval, &minid, &maxid);
//	}
//
//	double err = sum(error.mul(error))[0];
//	std::cout << "误差平方和 error：" << err << "\n";
//	cudaFree(initcuda);
//}

