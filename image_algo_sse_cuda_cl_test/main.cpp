#include "test_guassianfilter.h"

int main()
{
	int*initcuda;
	cudaMalloc((void**)&initcuda, 1);//ÿ���߳��״�ʹ��cudaʱ�����г�ʼ��һ�Σ����Խ�Լ���������ڴ�ռ�ʱ��
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	TEST_GUASSIAN_FILTER::Run();

	cudaFree(initcuda);
	system("pause");
	return 0;
}