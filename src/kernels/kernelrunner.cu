#include <random>
// zeros are padded by assuming spilled values to be zero
// stride is assumed to be 1 for simplicity
// kernel is always odd square matrix
void run_convolution(const char* kernel_name, void (*invoke_kernel)(float *, int, int, float*, int, int, float*), float *d_input, int M, int N, float *d_kernel, int m, int n, float *d_op, float *d_bench_op, std::mt19937 gen, int warmup_runs, int measurement_runs){

}