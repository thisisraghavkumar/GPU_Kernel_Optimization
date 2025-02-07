#inlcude <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <nvtx3/nvToolsExt.h>

#include "./mykernels.cuh"

void invoke_cudnn_conv(float *d_inp, int M, int N, float *d_kernel, int m, int n, float *d_out, float &elapsed_time, float *h_output=NULL, bool copy_output = false, int measurement_iterations=1, int warmup_iterations=1) {
    cudaHandle_t handle;
    cudaEvent_t beg, end;

    cudnnCreate(&handle);
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    cudnnTensorDescriptor_t inputDescriptor, outputDescriptor;
    cudnnFilterDescriptor_t kernelDescriptor;
    cudnnConvolutionDescriptor_t convolutionDescriptor;

    cudnnCreateTensorDescriptor(&inputDescriptor);
    cudnnCreateTensorDescriptor(&outputDescriptor);
    cudnnCreateFilterDescriptor(&kernelDescriptor);
    cudnnCreateConvolutionDescriptor(&convolutionDescriptor);

    int batchSize = 1, channels=1;
    int pad_m = m/2;
    int pad_n = n/2;
    int strid_m = 1, stride_n = 1;
    int dilation_m = 1, dilantion_n = 1;

    cudnnSetTensor4dDescriptor(
        inputDescriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize, channels, M, N
    );

    cudnnSetFilter4dDescriptor(
        kernelDescriptor,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        1,1,m,n
    );

    cudnnSetConvolution2dDescriptor(
        convolutionDescriptor,
        pad_m,pad_n,
        stride_m,stride_n,
        dilation_m,dilation_n,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    );

    int out_M, out_N;
    cudnnGetConvolution2dForwardOutputDim(
        convolutionDescriptor,
        inputDescriptor,
        kernelDescriptor,
        &batchSize, &channels,
        &out_M, &out_N
    );

    cudnnSetTensor4dDescriptor(
        outputDescriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize, channels, out_M, out_N
    );

    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm(
        handle,
        inputDescriptor,
        kernelDescriptor,
        convolutionDescriptor,
        outputDescriptor,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0, &algo
    );

    size_t workspaceBytes = 0;
    cudnnGetConvolutionForwardWorkspaceSize(handle,
        inputDescriptor,
        kernelDescriptor,
        convolutionDescriptor,
        outputDescriptor,
        algo,
        &workspaceBytes
    );

    float *d_workspace = NULL;
    if(workspaceBytes > 0) MCC(cudaMalloc(&d_workspace, workspaceBytes));

    float alpha=1.0f, beta=0.0f;

    cudnnConvolutionForward(handle, &alpha, inputDescriptor, d_inp, 
        kernelDescriptor, d_kernel, convolutionDescriptor, algo,
        d_workspace, workspaceBytes, &beta, outputDescriptor, d_out
    );

    if(copy_output && h_output != NULL) MMC(cudaMemcpy(h_output, d_out, sizeof(float)*out_M*out_N, cudaMemcpyDeviceToHost));

    for(int i=0 ;i<warmup_iterations-1;i++){
        cudnnConvolutionForward(handle, &alpha, inputDescriptor, d_inp, 
            kernelDescriptor, d_kernel, convolutionDescriptor, algo,
            d_workspace, workspaceBytes, &beta, outputDescriptor, d_out
        );
        cudaDeviceSynchronize();
    }

    cudaEventRecord(beg);
    nvtxRangePush("CUDNN kernel");
    for(int i=0; i<measurement_iterations; i++){
        cudnnConvolutionForward(handle, &alpha, inputDescriptor, d_inp, 
            kernelDescriptor, d_kernel, convolutionDescriptor, algo,
            d_workspace, workspaceBytes, &beta, outputDescriptor, d_out
        );
        cudaDeviceSynchronize();
    }
    nvtxRangePop();
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);

    if (workspaceBytes > 0) MMC(cudaFree(d_workspace));
    MMC(cudnnDestroyTensorDescriptor(inputDescriptor));
    MMC(cudnnDestroyTensorDescriptor(outputDescriptor));
    MMC(cudnnDestroyFilterDescriptor(kernelDescriptor));
    MMC(cudnnDestroyConvolutionDescriptor(convolutionDescriptor));
    cudnnDestroy(handle);
}