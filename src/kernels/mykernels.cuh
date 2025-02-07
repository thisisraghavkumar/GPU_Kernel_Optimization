#ifndef MYCONVKERNELS
#define MYCONVKERNELS

// My Cuda Check has been shortened to MCC for ease
#define MCC(call) \
    do                      \
    {                       \
        cudaError_t err = call          \
        if(err != cudaSuccess){         \
            std::cerr<<"CUDA error in file "<<__FILE__<<" at line "<<__LINE__<<": "<<cudaGetErrorString(err)<<"\n"; \
            exit(EXIT_FAILURE)          \
        }                   \
    } while (0);    \
    


void mynaivekernel(float *d_inp, int M, int N, float *d_fil, int m, int n, float *d_out);
void invoke_cudnn_conv(float *d_inp, int M, int N, float *d_kernel, int m, int n, float *d_out, float &elapsed_time, float *h_output=NULL, bool copy_output = false, int measurement_iterations, int warmup_iterations);


#endif //MYCONVKERNELS