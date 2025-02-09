#ifndef MYCONVKERNELS
#define MYCONVKERNELS

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <nvtx3/nvToolsExt.h>
#include <random>

#define INPROW 2048LL
#define INPCOL 2048LL
#define KERROW 5LL
#define KERCOL 5LL

#define BLOCK_SIZE 32
#define WARP_SIZE 32
#define sF sizeof(float)

#define CEILDIV(dividend, divisor) ((dividend + divisor - 1) / divisor)

__constant__ float d_kernel_const[KERROW * KERCOL]

// My Cuda Check has been shortened to MCC for ease
#define MCC(call) \
    do                      \
    {                       \
        cudaError_t err = call;          \
        if(err != cudaSuccess){         \
            std::cerr<<"CUDA error in file "<<__FILE__<<" at line "<<__LINE__<<": "<<cudaGetErrorString(err)<<"\n"; \
            exit(EXIT_FAILURE);          \
        }                   \
    } while (0);    \

void invoke_cudnn_conv(float *d_inp, int M, int N, float *d_kernel, int m, int n, float *d_out, float *elapsed_time, float *h_output, bool copy_output, int measurement_iterations, int warmup_iterations);
void invoke_mynaivekernel(float *d_inp, int M, int N, float *d_kernel, int m, int n, float *d_out, bool useConstantKernel=false);

void run_kernel(char *kernel_name, void (*invoke_kernel)(float*, int, int, float*, int, int, float *),float *dinp, int M, int N, float *dker, int m, int n, float *dout, float *hout, float *hout_ref, float *elapsed_time, std::mt19937 gen, bool useKernelFromConstants = false, int warmup_runs=1, int measurement_runs=50);
#endif //MYCONVKERNELS