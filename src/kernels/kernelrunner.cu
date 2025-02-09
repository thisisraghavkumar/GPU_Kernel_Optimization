#include "mykernels.cuh"

// zeros are padded by assuming spilled values to be zero
// stride is assumed to be 1 for simplicity
// kernel is always odd square matrix
void run_kernel(char *kernel_name, void (*invoke_kernel)(float*, int, int, float*, int, int, float *),float *dinp, int M, int N, float *dker, int m, int n, float *dout, float *hout, float *hout_ref, float &elapsed_time, std::mt19937 gen, bool useKernelFromConstants = false, int warmup_runs=1, int measurement_runs=50){
    cudaEvent_t beg, end;
    MCC(cudaEventCreate(&beg));
    MCC(cudaEventCreate(&end));
    int sizeC = M * N;
    invoke_kernel(dinp, M, N, dker, m, n, dout);
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
        	return 1;
    	}
    }
    for(int i=0; i<warmup_runs-1; i++){
        invoke_kernel(d_A, d_B, d_C, m, k, n);
    }
    MCC(cudaEventRecord(beg));
    nvtxRangePush(kernel_name);
    for(int i=0; i<measurement_runs; i++){
        invoke_kernel(d_A, d_B, d_C, m, k, n);
        cudaDeviceSynchronize();
    }
    nvtxRangePop();
    MCC(cudaEventRecord(end));
    MCC(cudaEventSynchronize(beg));
    MCC(cudaEventSynchronize(end));
    MCC(cudaEventElapsedTime(&elapsed_time, beg, end));
    return elapsed_time;
}