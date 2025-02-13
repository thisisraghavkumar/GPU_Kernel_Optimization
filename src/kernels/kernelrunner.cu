#include "mykernels.cuh"

// zeros are padded by assuming spilled values to be zero
// stride is assumed to be 1 for simplicity
// kernel is always odd square matrix
void run_kernel(const char *kernel_name, void (*invoke_kernel)(float*, int, int, float*, const int, const int, float *, bool),float *dinp, int M, int N, float *dker, const int m, const int n, float *dout, float *hout, float *hout_ref, float *elapsed_time, std::mt19937 gen, bool useKernelFromConstants, int warmup_runs, int measurement_runs){
    cudaEvent_t beg, end;
    MCC(cudaEventCreate(&beg));
    MCC(cudaEventCreate(&end));
    int sizeC = M * N;
    invoke_kernel(dinp, M, N, dker, m, n, dout, useKernelFromConstants);
    cudaError_t err = cudaGetLastError();  
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Force synchronization to catch errors that might happen asynchronously
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        std::cerr << "CUDA kernel execution error: " << cudaGetErrorString(syncErr) << std::endl;
        return;
    }
    MCC(cudaMemcpy(hout, dout, sizeof(float)*sizeC, cudaMemcpyDeviceToHost));
    for(int i=0; i < 500; i++){
    	int randomRow = gen() % M;
    	int randomCol = gen() % N;
    	float tolerance = 20;
    	if(fabs(hout[randomRow * n + randomCol] - hout_ref[randomRow * n + randomCol]) > tolerance){
        	std::cout <<"For kernel "<<kernel_name<<std::endl;
        	std::cout <<"Error: Reference and my kernel results do not match at "<<randomRow<<", "<<randomCol << std::endl;
        	std::cout <<"Content of hout = "<<std::setprecision(32)<<hout[randomRow * n + randomCol]<<std::endl;
        	std::cout <<"Content of hout_ref = "<<std::setprecision(32)<<hout_ref[randomRow * n + randomCol]<<std::endl;
    	}
    }
    for(int i=0; i<warmup_runs-1; i++){
        invoke_kernel(dinp, M, N, dker, m, n, dout, useKernelFromConstants);
        cudaError_t err = cudaGetLastError();  
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        // Force synchronization to catch errors that might happen asynchronously
        cudaError_t syncErr = cudaDeviceSynchronize();
        if (syncErr != cudaSuccess) {
            std::cerr << "CUDA kernel execution error: " << cudaGetErrorString(syncErr) << std::endl;
            return;
        }
    }
    MCC(cudaEventRecord(beg));
    nvtxRangePush(kernel_name);
    for(int i=0; i<measurement_runs; i++){
        invoke_kernel(dinp, M, N, dker, m, n, dout, useKernelFromConstants);
        cudaError_t err = cudaGetLastError();  
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        // Force synchronization to catch errors that might happen asynchronously
        cudaError_t syncErr = cudaDeviceSynchronize();
        if (syncErr != cudaSuccess) {
            std::cerr << "CUDA kernel execution error: " << cudaGetErrorString(syncErr) << std::endl;
            return;
        }
    }
    nvtxRangePop();
    MCC(cudaEventRecord(end));
    MCC(cudaEventSynchronize(beg));
    MCC(cudaEventSynchronize(end));
    MCC(cudaEventElapsedTime(elapsed_time, beg, end));
}