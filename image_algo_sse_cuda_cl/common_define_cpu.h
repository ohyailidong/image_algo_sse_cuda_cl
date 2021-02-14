#ifndef COMM0N_DEFINE_H
#define COMMON_DEFINE_H

#ifdef DLL_EXPORT
#define EXPORT_IMAGE_ALGO_DLL _declspec(dllexport)
#else 
#define EXPORT_IMAGE_ALGO_DLL
#endif

#include <iostream>
#include <algorithm>
#include <assert.h>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#define CVLIB_NAMESPACE_BEGIN namespace cvlib{
#define CVLIB_NAMESPACE_END }
#define CUDA_NAMESPACE_BEGIN namespace cuda{
#define CUDA_NAMESPACE_END }
#define MULTI_THREAD_NAMESPACE_BEGIN namespace mt{
#define MULTI_THREAD_NAMESPACE_END }

#define CPU_FUN_NMAESPACE_BEGIN namespace cpu{
#define CPU_FUN_NMAESPACE_END }

CPU_FUN_NMAESPACE_BEGIN
int borderInterpolateDefault(int p, int len);
CPU_FUN_NMAESPACE_END


#endif