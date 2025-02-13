#include "./kernels/mykernels.cuh"


void printMatrix(const char *heading, const float* matrix, int m, int n) {
    const int width = 10;      // Column width
    const int precision = 4;   // Decimal precision

    std::cout<<heading<<":\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << std::setw(width) << std::fixed << std::setprecision(precision)
                      << matrix[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
}

/*
 * Function to populate an array of floats with random values
 */
void populate_array(float *arr, int size, std::mt19937 &gen, std::uniform_real_distribution<float> &dis)
{
    for (int i = 0; i < size; i++)
    {
        arr[i] = dis(gen);
    }
}

void CudaDeviceInfo()
{
    int deviceId;

    cudaGetDevice(&deviceId);

    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, deviceId);
    std::cout << "Device ID                         : " << deviceId << std::endl;
    std::cout << "Name                              : " << props.name << std::endl;
    std::cout << "Compute Capability                : " << props.major << "." << props.minor << std::endl;
    std::cout << "Memory Bus Width                  : " << props.memoryBusWidth << std::endl;
    std::cout << "Max threads per block             : " << props.maxThreadsPerBlock << std::endl;
    std::cout << "Max threads per multi-processor   : " << props.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Registers per block               : " << props.regsPerBlock << std::endl;
    std::cout << "Registers per multi-processor     : " << props.regsPerMultiprocessor << std::endl;
    std::cout << "Total Global Memory               : " << props.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
    std::cout << "Shared Memory per block           : " << props.sharedMemPerBlock / 1024 << "KB" << std::endl;
    std::cout << "Shared Memory per multi-processor : " << props.sharedMemPerMultiprocessor / 1024 << "KB" << std::endl;
    std::cout << "Total Constant Memory             : " << props.totalConstMem / 1024 << "KB" << std::endl;
    std::cout << "Multi-processor count             : " << props.multiProcessorCount << std::endl;
    std::cout << "Warp Size                         : " << props.warpSize << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
}

int main(){
    CudaDeviceInfo();

    int M = INPROW; // input height
    int N = INPCOL; // input width
    const int m = KERROW;    // kernel height
    const int n = KERCOL;    // kernel width

    int input_size = M*N;
    int kernel_size = m*n;

    int warmpup_runs = 5;
    int measurement_runs = 50;
    long long numoperations = M*N*(2*m*n - 1);
    float *h_inp, *h_fil, *h_out, *h_out_ref;
    float *d_inp, *d_fil, *d_out;

    float elapsed_time;
    cudaEvent_t beg, end;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(50.0, 25.0);

    MCC(cudaMalloc(&d_inp, sizeof(float)*input_size));
    MCC(cudaMalloc(&d_fil, sizeof(float)*kernel_size)); // maybe I should use constant space in GPU for small kernels
    MCC(cudaMalloc(&d_out, sizeof(float)*input_size));

    h_inp = new float[input_size];
    h_fil = new float[kernel_size];
    h_out = new float[input_size];
    h_out_ref = new float[input_size];
    populate_array(h_inp, input_size, gen, dis);
    //printMatrix("Input",h_inp, M, N);
    populate_array(h_fil, kernel_size, gen, dis);
    //printMatrix("Kernel", h_fil, m, n);
    MCC(cudaMemcpy(d_inp, h_inp, sizeof(float)*input_size, cudaMemcpyHostToDevice));
    MCC(cudaMemcpy(d_fil, h_fil, sizeof(float)*kernel_size, cudaMemcpyHostToDevice)); // maybe this will change if I choose to use constant space for small kernels
    //MCC(cudaMemcpyToSymbol(d_kernel_const, h_fil, sizeof(float)*kernel_size));

    invoke_mynaivekernel(d_inp, M, N, d_fil, m, n, d_out, false);
    MCC(cudaMemcpy(h_out_ref, d_out, sizeof(float)*input_size, cudaMemcpyDeviceToHost));
    //printMatrix("Output with kernel in memory", h_out_ref, M, N);
    //invoke_mynaivekernel(d_inp, M, N, d_fil, m, n, d_out, true);
    //MCC(cudaMemcpy(h_out_ref, d_out, sizeof(float)*input_size, cudaMemcpyDeviceToHost));
    //printMatrix("Output with kernel as constant",h_out_ref, M, N);

    auto printRow = [](const std::string &name, float time, long long ops, int runs)
    {
        float avg_time = time / runs;
        float gflops = (ops / (avg_time / 1000.0f)) / 1e9;
        std::cout << std::setw(20) << std::left << name
                  << std::setw(20) << avg_time
                  << std::setw(20) << gflops << std::endl;
    };

    std::cout << std::fixed << std::setprecision(5);
    std::cout << "Number of operations: " << numoperations << std::endl;
    std::cout << std::fixed << std::setprecision(5); // Set decimal precision for floats

    // Print the table header
    std::cout << std::setw(20) << std::left << "Kernel Name"
              << std::setw(20) << "Time Taken (ms)"
              << std::setw(20) << "GFLOP/S" << std::endl;

    std::cout << std::string(60, '-') << std::endl;
    run_kernel("Naive kernel in memory", invoke_mynaivekernel, d_inp,M,N, d_fil, m, n, d_out, h_out, h_out_ref, &elapsed_time, gen, false, 5, measurement_runs);
    printRow("Naive kernel", elapsed_time, numoperations, measurement_runs);
    run_kernel("Shared memory kernel 1 with kernel in memory", invoke_mysharedmemkernel1, d_inp,M,N, d_fil, m, n, d_out, h_out, h_out_ref, &elapsed_time, gen, false, 5, measurement_runs);
    printRow("Shared memory kernel 1 with kernel in memory", elapsed_time, numoperations, measurement_runs);

    MCC(cudaFree(d_inp));
    MCC(cudaFree(d_fil));
    MCC(cudaFree(d_out));
    free(h_inp);
    free(h_fil);
    free(h_out);
    free(h_out_ref);
}
