#include "test_guassianfilter.h"
#include "test_boxfilter.h"
#include "test_medianblur.h"
#include "test_bilateralfilter.h"
#include "test_histogram.h"
#include "test_copymakeborder.h"
#include "test_matchtemplate.h"
int main()
{
	int*initcuda;
	cudaMalloc((void**)&initcuda, 1);//ÿ���߳��״�ʹ��cudaʱ�����г�ʼ��һ�Σ����Խ�Լ���������ڴ�ռ�ʱ��
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	TEST_COPYMAKEBORDER::Run();
	TEST_BOX_FILTER::Run();
	TEST_MEDIANBLUR_FILTER::Run();
	TEST_BILATERAL_FILTER::Run();
	TEST_GUASSIAN_FILTER::Run();
	TEST_HISTOGRAM::Run();
	TEST_INTEGRAL::Run();
	TEST_MATCH_TEMPLATE::Run();
	int a;
	int b;
	cudaFree(initcuda);
	system("pause");
	return 0;
}