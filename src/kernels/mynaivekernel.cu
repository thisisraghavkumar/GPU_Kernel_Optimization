#include "mykernels.cuh"

__global__ void mynaive_convolution(float *dinp, int M, int N, float *dker, int m, int n, float *dout){
    int opRow = blockIdx.y * blockDim.y + threadIdx.y;
    int opCol = blockIdx.x * blockDim.x + threadIdx.x;

    if(opRow < M && opCol < N){
        float op = 0.0;
        int limH = m/2;
        int limW = n/2;
        for(int i = -1 * limH; i <= limH; ++i){
            for(int j = -1 * limW; j <= limW; ++j){
                int kerRow = limH + i;
                int kerCol = limW + j;
                int kerIdx = kerRow * n + kerCol;
                int inpRow = opRow + i;
                int inpCol = opCol + j;
                int inpIdx = inpRow * N + inpCol;
                if(inpRow >= 0 && inpCol >= 0){
                    op += dinp[inpIdx] * dker[kerIdx];
                }
            }
        }
        dout[opRow * N + opCol] = op;
    }
}

/*
__global__ void mynaive_convolution_const(float *dinp, int M, int N, int m, int n, float *dout){
    int opRow = blockIdx.y * blockDim.y + threadIdx.y;
    int opCol = blockIdx.x * blockDim.x + threadIdx.x;
    if(opRow < M && opCol < N){
        float op = 0.0;
        int limH = m/2;
        int limW = n/2;
        for(int i = -1 * limH; i <= limH; ++i){
            for(int j = -1 * limW; j <= limW; ++j){
                int kerRow = limH + i;
                int kerCol = limW + j;
                int kerIdx = kerRow * n + kerCol;
                int inpRow = opRow + i;
                int inpCol = opCol + j;
                int inpIdx = inpRow * N + inpCol;
                if(inpRow >= 0 && inpCol >= 0){
                    op += dinp[inpIdx] * d_kernel_const[kerIdx];
                }
            }
        }
        dout[opRow * N + opCol] = op;
    }
}
*/

void invoke_mynaivekernel(float *dinp, int M, int N, float *dfil, int m, int n, float *dout, bool useConstantKernel){
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(CEILDIV(N,BLOCK_SIZE),CEILDIV(M,BLOCK_SIZE));
    mynaive_convolution<<<gridDim,blockDim>>>(dinp,M,N,dfil,m,n,dout);
}