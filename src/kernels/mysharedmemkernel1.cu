#include "mykernels.cuh"

// x and y directions of blocks correspond to that of image
// boundary cells are included in shared memory
// the kernel is smaller than block 32 x 32
template <const int m, const int n>
__global__ void mysharedmemkernel1(float *dinp, int M, int N, float *dker, float *dout){
    int opRow = blockIdx.y * blockDim.y + threadIdx.y;
    int opCol = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float imSlice[(BLOCK_SIZE+m-1)*(BLOCK_SIZE+n-1)];
    __shared__ float sharedKernel[m*n];
    int sharedRow = threadIdx.y + (m/2);
    int sharedCol = threadIdx.x + (n/2);
    // coordinate of a cell in imSlice and dinp will be different
    // the relation is imSlice_x = dinp_x + n/2 and imSlice_y = dinp_y + m/2
    // if thread lies inside the image
    if(opRow < M && opCol < N){
        //the top left cell will load the top left corner of the halo
        if(threadIdx.x == 0 && threadIdx.y == 0){
            for(int i=-1*(m/2); i <= 0; i++){
                for(int j=-1*(n/2); j <= 0; j++){
                    if(opCol + j >= 0 && opCol + j < N && opRow + i >= 0 && opRow + i < M){
                        imSlice[(sharedRow + i) * (blockDim.x+n-1) + (sharedCol + j)] = dinp[(opRow + i) * N + (opCol + j)];
                    }else{
                        imSlice[(sharedRow + i) * (blockDim.x+n-1) + (sharedCol + j)] = 0.0f;
                    }
                }
            }
        } else if(threadIdx.x == blockDim.x-1 && threadIdx.y == 0){
            //the topright thread will load the top right corner of the halo
            for(int i=-1*(m/2); i <= 0; i++){
                for(int j=0; j <= n/2; j++){
                    if(opCol + j >= 0 && opCol + j < N && opRow + i >= 0 && opRow + i < M){
                        imSlice[(sharedRow + i) * (blockDim.x+n-1) + (sharedCol + j)] = dinp[(opRow + i) * N + (opCol + j)];
                    }else{
                        imSlice[(sharedRow + i) * (blockDim.x+n-1) + (sharedCol + j)] = 0.0f;
                    }
                }
            }
        } else if(threadIdx.x == 0 && threadIdx.y == blockDim.y-1){
            //the bottom left thread will load the bottom lefth corner of the halo
            for(int i=0; i <= m/2; i++){
                for(int j=-1*(n/2); j <= 0; j++){
                    if(opCol + j >= 0 && opCol + j < N && opRow + i >= 0 && opRow + i < M){
                        imSlice[(sharedRow + i) * (blockDim.x+n-1) + (sharedCol + j)] = dinp[(opRow + i) * N + (opCol + j)];
                    }else{
                        imSlice[(sharedRow + i) * (blockDim.x+n-1) + (sharedCol + j)] = 0.0f;
                    }
                }
            }
        } else if(threadIdx.x == blockDim.x-1 && threadIdx.y == blockDim.y-1){
            //the bottom right thread will load the bottom right corner of the halo
            for(int i=0; i <= m/2; i++){
                for(int j=0; j <= n/2; j++){
                    if(opCol + j >= 0 && opCol + j < N && opRow + i >= 0 && opRow + i < M){
                        imSlice[(sharedRow + i) * (blockDim.x+n-1) + (sharedCol + j)] = dinp[(opRow + i) * N + (opCol + j)];
                    }else{
                        imSlice[(sharedRow + i) * (blockDim.x+n-1) + (sharedCol + j)] = 0.0f;
                    }
                }
            }
        } else if(threadIdx.x == 0) {
            // thread on left edge will load a line to its left
            for(int j = -1*(n/2); j <= 0; j++){
                if(opCol + j >= 0 && opCol + j < N){
                    imSlice[sharedRow * (blockDim.x+n-1) + (sharedCol + j)] = dinp[opRow * N + (opCol + j)];
                } else {
                    imSlice[sharedRow * (blockDim.x+n-1) + (sharedCol + j)] = 0.0f;
                }
            }
        } else if(threadIdx.x == blockDim.x-1) {
            // thread on right edge will load a line to its left
            for(int j = 0; j <= n/2; j++){
                if(opCol + j >= 0 && opCol + j < N){
                    imSlice[sharedRow * (blockDim.x+n-1) + (sharedCol + j)] = dinp[opRow * N + (opCol + j)];
                } else {
                    imSlice[sharedRow * (blockDim.x+n-1) + (sharedCol + j)] = 0.0f;
                }
            }
        } else if(threadIdx.y == 0) {
            // thread on top edge will load a line to its up
            for(int j = -1*(m/2); j <= 0; j++){
                if(opRow + j >= 0 && opRow + j < M){
                    imSlice[(sharedRow+j) * (blockDim.x+n-1) + sharedCol] = dinp[(opRow+j) * N + opCol];
                } else {
                    imSlice[(sharedRow+j) * (blockDim.x+n-1) + sharedCol] = 0.0f;
                }
            }
        } else if(threadIdx.y == blockDim.y-1) {
            // thread on bottom edge will load a line to its bottom
            for(int j = 0; j <= m/2; j++){
                if(opRow + j >= 0 && opRow + j < M){
                    imSlice[(sharedRow+j) * (blockDim.x+n-1) + sharedCol] = dinp[(opRow+j) * N + opCol];
                } else {
                    imSlice[(sharedRow+j) * (blockDim.x+n-1) + sharedCol] = 0.0f;
                }
            }
        } else {
            //every thread loads the cell it is centered on
            imSlice[sharedRow * (blockDim.x+n-1) + sharedCol] = dinp[opRow * N + opCol];
        }
        if(threadIdx.x < m && threadIdx.y < n){
            sharedKernel[threadIdx.y * n + threadIdx.x] = dker[threadIdx.y * n + threadIdx.x];
        }
        __syncthreads();
        float sum = 0.0f;
        for(int i=-1 * (m/2); i <= m/2; i++){
            for(int j = -1 * (n/2); j <= (n/2); j++){
                sum += imSlice[(sharedRow+i) * (blockDim.x + n - 1) + (sharedCol+j)] * sharedKernel[((m/2)+i)*n + (n/2+j)];
            }
        }
        dout[opRow*N+opCol] = sum;
    }
}

void invoke_mysharedmemkernel1(float *dinp, int M, int N, float *dker, const int m, const int n, float *dout, bool useConstantKernel){
    dim3 gridSize(CEILDIV(N, BLOCK_SIZE), CEILDIV(M, BLOCK_SIZE));
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    mysharedmemkernel1<KERROW,KERCOL><<<gridSize,blockSize>>>(dinp, M, N, dker, dout);
}