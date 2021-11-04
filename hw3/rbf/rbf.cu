# include <math.h>
# include <time.h>
# include <stdio.h>
# include <stdlib.h>
# include <iostream>
# include <sys/time.h>

# include "cuda_runtime.h"

using namespace std;

const int DIM = 1024, AS = 32, BS = 32;
const float sigma = 1.0;
const bool PRINT_RESULT = false;

/*
RBF Kernel Implementation on CPU.
Param @ sigma: the parameter sigma of RBF kernel;
      @ a, b, c: the input arrays and the output array;
      @ dim, as, bs: the size of the array, a should be of size as * dim, and b should be of size bs * dim.
*/
void RBF_CPU(float sigma, float a[][DIM], float b[][DIM], float c[], int dim, int as, int bs) {
    for (int i = 0; i < as; ++ i)
        for (int j = 0; j < bs; ++ j) {
            float Pvalue = 0.0;
            for (int k = 0; k < dim; ++ k)
                Pvalue += (a[i][k] - b[j][k]) * (a[i][k] - b[j][k]);
            c[i * bs + j] = exp(- Pvalue / (2 * sigma * sigma));
        }
}

/* 
Simple RBF Kernel Implementation.
Param @ sigma: the parameter sigma of RBF kernel;
      @ Md, Nd, Pd: the input values and the output array;
      @ dim: the dimension of features;
      @ Pdim: the width of the result array.
*/
__global__ void RBFKernel_Simple(float sigma, float *Md, float *Nd, float *Pd, int dim, int Pdim) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float Pvalue = 0.0;

    for (int k = 0; k < dim; ++ k) {
        float Mds = Md[tx * dim + k];
        float Nds = Nd[ty * dim + k];
        Pvalue += (Mds - Nds) * (Mds - Nds);
    }
    Pd[tx * Pdim + ty] = exp(- Pvalue / (2 * sigma * sigma));
}

/*
Wrapper function of Simple RBF Kernel Implementation.
Param @ sigma: the parameter sigma of RBF kernel;
      @ a, b, c: the input arrays and the output array;
      @ dim, as, bs: the size of the array, a should be of size as * dim, and b should be of size bs * dim.
Return @ the elapsed time of GPU calculation.
*/
float RBF_CUDA_Simple(float sigma, float a[][DIM], float b[][DIM], float c[], int dim, int as, int bs) {
    cudaEvent_t start, stop;
    float elapsedTime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block(as, bs);
    float *M, *N, *P;
    cudaMalloc((void **)&M, as * dim * sizeof(float));
    cudaMalloc((void **)&N, bs * dim * sizeof(float));
    cudaMalloc((void **)&P, as * bs * sizeof(float));

    cudaMemcpy(M, a, as * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(N, b, bs * dim * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);
    RBFKernel_Simple <<<1, block>>> (sigma, M, N, P, dim, bs);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(c, P, as * bs * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(M);
    cudaFree(N);
    cudaFree(P);

    return elapsedTime;
}

/* 
RBF Kernel Implementation based on tilling (width 2).
Param @ sigma: the parameter sigma of RBF kernel;
      @ Md, Nd, Pd: the input values and the output array;
      @ dim: the dimension of features;
      @ Pdim: the width of the result array.
*/
const int TILE_WIDTH_2 = 2;
__global__ void RBFKernel_Tilling_2(float sigma, float *Md, float *Nd, float *Pd, int dim, int Pdim) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float Pvalue = 0.0;
    int Mrow = bx * TILE_WIDTH_2 + tx;
    int Nrow = by * TILE_WIDTH_2 + ty;
    __shared__ float Mds[TILE_WIDTH_2][TILE_WIDTH_2];
    __shared__ float Nds[TILE_WIDTH_2][TILE_WIDTH_2];

    for (int i = 0; i < dim / TILE_WIDTH_2; ++ i) {
        Mds[tx][ty] = Md[Mrow * dim + i * TILE_WIDTH_2 + ty];
        Nds[ty][tx] = Nd[Nrow * dim + i * TILE_WIDTH_2 + tx];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH_2; ++ k)
            Pvalue += (Mds[tx][k] - Nds[ty][k]) * (Mds[tx][k] - Nds[ty][k]);
        __syncthreads();
    }
    Pd[Mrow * Pdim + Nrow] = exp(- Pvalue / (2 * sigma * sigma));
}

/*
Wrapper function of RBF Kernel Implementation based on tilling (width 2).
Param @ sigma: the parameter sigma of RBF kernel;
      @ a, b, c: the input arrays and the output array;
      @ dim, as, bs: the size of the array, a should be of size as * dim, and b should be of size bs * dim.
Return @ the elapsed time of GPU calculation.
*/
float RBF_CUDA_Tilling_2(float sigma, float a[][DIM], float b[][DIM], float c[], int dim, int as, int bs) {
    cudaEvent_t start, stop;
    float elapsedTime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 grid(as / TILE_WIDTH_2, bs / TILE_WIDTH_2);
    dim3 block(TILE_WIDTH_2, TILE_WIDTH_2);
    float *M, *N, *P;
    cudaMalloc((void **)&M, as * dim * sizeof(float));
    cudaMalloc((void **)&N, bs * dim * sizeof(float));
    cudaMalloc((void **)&P, as * bs * sizeof(float));

    cudaMemcpy(M, a, as * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(N, b, bs * dim * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);
    RBFKernel_Tilling_2 <<<grid, block>>> (sigma, M, N, P, dim, bs);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(c, P, as * bs * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(M);
    cudaFree(N);
    cudaFree(P);

    return elapsedTime;
}

/* 
RBF Kernel Implementation based on tilling (width 4).
Param @ sigma: the parameter sigma of RBF kernel;
      @ Md, Nd, Pd: the input values and the output array;
      @ dim: the dimension of features;
      @ Pdim: the width of the result array.
*/
const int TILE_WIDTH_4 = 4;
__global__ void RBFKernel_Tilling_4(float sigma, float *Md, float *Nd, float *Pd, int dim, int Pdim) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float Pvalue = 0.0;
    int Mrow = bx * TILE_WIDTH_4 + tx;
    int Nrow = by * TILE_WIDTH_4 + ty;
    __shared__ float Mds[TILE_WIDTH_4][TILE_WIDTH_4];
    __shared__ float Nds[TILE_WIDTH_4][TILE_WIDTH_4];

    for (int i = 0; i < dim / TILE_WIDTH_4; ++ i) {
        Mds[tx][ty] = Md[Mrow * dim + i * TILE_WIDTH_4 + ty];
        Nds[ty][tx] = Nd[Nrow * dim + i * TILE_WIDTH_4 + tx];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH_4; ++ k)
            Pvalue += (Mds[tx][k] - Nds[ty][k]) * (Mds[tx][k] - Nds[ty][k]);
        __syncthreads();
    }
    Pd[Mrow * Pdim + Nrow] = exp(- Pvalue / (2 * sigma * sigma));
}

/*
Wrapper function of RBF Kernel Implementation based on tilling (width 4).
Param @ sigma: the parameter sigma of RBF kernel;
      @ a, b, c: the input arrays and the output array;
      @ dim, as, bs: the size of the array, a should be of size as * dim, and b should be of size bs * dim.
Return @ the elapsed time of GPU calculation.
*/
float RBF_CUDA_Tilling_4(float sigma, float a[][DIM], float b[][DIM], float c[], int dim, int as, int bs) {
    cudaEvent_t start, stop;
    float elapsedTime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 grid(as / TILE_WIDTH_4, bs / TILE_WIDTH_4);
    dim3 block(TILE_WIDTH_4, TILE_WIDTH_4);
    float *M, *N, *P;
    cudaMalloc((void **)&M, as * dim * sizeof(float));
    cudaMalloc((void **)&N, bs * dim * sizeof(float));
    cudaMalloc((void **)&P, as * bs * sizeof(float));

    cudaMemcpy(M, a, as * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(N, b, bs * dim * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);
    RBFKernel_Tilling_4 <<<grid, block>>> (sigma, M, N, P, dim, bs);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(c, P, as * bs * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(M);
    cudaFree(N);
    cudaFree(P);

    return elapsedTime;
}

/* 
RBF Kernel Implementation based on tilling (width 8).
Param @ sigma: the parameter sigma of RBF kernel;
      @ Md, Nd, Pd: the input values and the output array;
      @ dim: the dimension of features;
      @ Pdim: the width of the result array.
*/
const int TILE_WIDTH_8 = 8;
__global__ void RBFKernel_Tilling_8(float sigma, float *Md, float *Nd, float *Pd, int dim, int Pdim) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float Pvalue = 0.0;
    int Mrow = bx * TILE_WIDTH_8 + tx;
    int Nrow = by * TILE_WIDTH_8 + ty;
    __shared__ float Mds[TILE_WIDTH_8][TILE_WIDTH_8];
    __shared__ float Nds[TILE_WIDTH_8][TILE_WIDTH_8];

    for (int i = 0; i < dim / TILE_WIDTH_8; ++ i) {
        Mds[tx][ty] = Md[Mrow * dim + i * TILE_WIDTH_8 + ty];
        Nds[ty][tx] = Nd[Nrow * dim + i * TILE_WIDTH_8 + tx];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH_8; ++ k)
            Pvalue += (Mds[tx][k] - Nds[ty][k]) * (Mds[tx][k] - Nds[ty][k]);
        __syncthreads();
    }
    Pd[Mrow * Pdim + Nrow] = exp(- Pvalue / (2 * sigma * sigma));
}

/*
Wrapper function of RBF Kernel Implementation based on tilling (width 8).
Param @ sigma: the parameter sigma of RBF kernel;
      @ a, b, c: the input arrays and the output array;
      @ dim, as, bs: the size of the array, a should be of size as * dim, and b should be of size bs * dim.
Return @ the elapsed time of GPU calculation.
*/
float RBF_CUDA_Tilling_8(float sigma, float a[][DIM], float b[][DIM], float c[], int dim, int as, int bs) {
    cudaEvent_t start, stop;
    float elapsedTime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 grid(as / TILE_WIDTH_8, bs / TILE_WIDTH_8);
    dim3 block(TILE_WIDTH_8, TILE_WIDTH_8);
    float *M, *N, *P;
    cudaMalloc((void **)&M, as * dim * sizeof(float));
    cudaMalloc((void **)&N, bs * dim * sizeof(float));
    cudaMalloc((void **)&P, as * bs * sizeof(float));

    cudaMemcpy(M, a, as * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(N, b, bs * dim * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);
    RBFKernel_Tilling_8 <<<grid, block>>> (sigma, M, N, P, dim, bs);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(c, P, as * bs * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(M);
    cudaFree(N);
    cudaFree(P);

    return elapsedTime;
}

/* 
RBF Kernel Implementation based on tilling (width 16).
Param @ sigma: the parameter sigma of RBF kernel;
      @ Md, Nd, Pd: the input values and the output array;
      @ dim: the dimension of features;
      @ Pdim: the width of the result array.
*/
const int TILE_WIDTH_16 = 16;
__global__ void RBFKernel_Tilling_16(float sigma, float *Md, float *Nd, float *Pd, int dim, int Pdim) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float Pvalue = 0.0;
    int Mrow = bx * TILE_WIDTH_16 + tx;
    int Nrow = by * TILE_WIDTH_16 + ty;
    __shared__ float Mds[TILE_WIDTH_16][TILE_WIDTH_16];
    __shared__ float Nds[TILE_WIDTH_16][TILE_WIDTH_16];

    for (int i = 0; i < dim / TILE_WIDTH_16; ++ i) {
        Mds[tx][ty] = Md[Mrow * dim + i * TILE_WIDTH_16 + ty];
        Nds[ty][tx] = Nd[Nrow * dim + i * TILE_WIDTH_16 + tx];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH_16; ++ k)
            Pvalue += (Mds[tx][k] - Nds[ty][k]) * (Mds[tx][k] - Nds[ty][k]);
        __syncthreads();
    }
    Pd[Mrow * Pdim + Nrow] = exp(- Pvalue / (2 * sigma * sigma));
}

/*
Wrapper function of RBF Kernel Implementation based on tilling (width 16).
Param @ sigma: the parameter sigma of RBF kernel;
      @ a, b, c: the input arrays and the output array;
      @ dim, as, bs: the size of the array, a should be of size as * dim, and b should be of size bs * dim.
Return @ the elapsed time of GPU calculation.
*/
float RBF_CUDA_Tilling_16(float sigma, float a[][DIM], float b[][DIM], float c[], int dim, int as, int bs) {
    cudaEvent_t start, stop;
    float elapsedTime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 grid(as / TILE_WIDTH_16, bs / TILE_WIDTH_16);
    dim3 block(TILE_WIDTH_16, TILE_WIDTH_16);
    float *M, *N, *P;
    cudaMalloc((void **)&M, as * dim * sizeof(float));
    cudaMalloc((void **)&N, bs * dim * sizeof(float));
    cudaMalloc((void **)&P, as * bs * sizeof(float));

    cudaMemcpy(M, a, as * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(N, b, bs * dim * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);
    RBFKernel_Tilling_16 <<<grid, block>>> (sigma, M, N, P, dim, bs);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(c, P, as * bs * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(M);
    cudaFree(N);
    cudaFree(P);

    return elapsedTime;
}

/* 
RBF Kernel Implementation based on tilling (width 32).
Param @ sigma: the parameter sigma of RBF kernel;
      @ Md, Nd, Pd: the input values and the output array;
      @ dim: the dimension of features;
      @ Pdim: the width of the result array.
*/
const int TILE_WIDTH_32 = 32;
__global__ void RBFKernel_Tilling_32(float sigma, float *Md, float *Nd, float *Pd, int dim, int Pdim) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float Pvalue = 0.0;
    int Mrow = bx * TILE_WIDTH_32 + tx;
    int Nrow = by * TILE_WIDTH_32 + ty;
    __shared__ float Mds[TILE_WIDTH_32][TILE_WIDTH_32];
    __shared__ float Nds[TILE_WIDTH_32][TILE_WIDTH_32];

    for (int i = 0; i < dim / TILE_WIDTH_32; ++ i) {
        Mds[tx][ty] = Md[Mrow * dim + i * TILE_WIDTH_32 + ty];
        Nds[ty][tx] = Nd[Nrow * dim + i * TILE_WIDTH_32 + tx];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH_32; ++ k)
            Pvalue += (Mds[tx][k] - Nds[ty][k]) * (Mds[tx][k] - Nds[ty][k]);
        __syncthreads();
    }
    Pd[Mrow * Pdim + Nrow] = exp(- Pvalue / (2 * sigma * sigma));
}

/*
Wrapper function of RBF Kernel Implementation based on tilling (width 32).
Param @ sigma: the parameter sigma of RBF kernel;
      @ a, b, c: the input arrays and the output array;
      @ dim, as, bs: the size of the array, a should be of size as * dim, and b should be of size bs * dim.
Return @ the elapsed time of GPU calculation.
*/
float RBF_CUDA_Tilling_32(float sigma, float a[][DIM], float b[][DIM], float c[], int dim, int as, int bs) {
    cudaEvent_t start, stop;
    float elapsedTime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 grid(as / TILE_WIDTH_32, bs / TILE_WIDTH_32);
    dim3 block(TILE_WIDTH_32, TILE_WIDTH_32);
    float *M, *N, *P;
    cudaMalloc((void **)&M, as * dim * sizeof(float));
    cudaMalloc((void **)&N, bs * dim * sizeof(float));
    cudaMalloc((void **)&P, as * bs * sizeof(float));

    cudaMemcpy(M, a, as * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(N, b, bs * dim * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);
    RBFKernel_Tilling_32 <<<grid, block>>> (sigma, M, N, P, dim, bs);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(c, P, as * bs * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(M);
    cudaFree(N);
    cudaFree(P);

    return elapsedTime;
}

// Print the result array.
void prtResult(float res[], int n, int m) {
    if (PRINT_RESULT) {
        printf("Result: \n");
        for (int i = 0; i < n; ++ i, printf("\n"))
            for (int j = 0; j < m; ++ j) 
                printf("%.2lf ", res[i * m + j]);
    } else {
        float sum = 0.0;
        for (int i = 0; i < n * m; ++ i)
            sum += res[i];
        printf("The sum of the matrix: %.2lf\n", sum);
    }
}

int main() {
    // Randomly generate array for testing.
    srand(time(0));
    float a[AS][DIM], b[BS][DIM];

    for (int i = 0; i < AS; ++ i)
        for (int j = 0; j < DIM; ++ j)
            a[i][j] = 1.0 / (rand() % 500 + 1.0);
    for (int i = 0; i < BS; ++ i)
        for (int j = 0; j < DIM; ++ j)
            b[i][j] = 1.0 / (rand() % 500 + 1.0);
    
    // ============================================================ //
    double duration; // execution time
    timeval start, end; // since clock() is not accurate, we use gettimeofday(...) in 'sys/time.h' to calculate running time.

    // => Baseline: CPU
    float c_CPU[AS * BS];
    gettimeofday(&start, 0);
    RBF_CPU(sigma, a, b, c_CPU, DIM, AS, BS);
    gettimeofday(&end, 0);
    duration = (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
    prtResult(c_CPU, AS, BS);
    printf("CPU method running time: %.2f us\n", duration);
    
    // ==> CUDA, Simple
    float c_GPU_simple[AS * BS];
    float elapsedTime_simple = RBF_CUDA_Simple(sigma, a, b, c_GPU_simple, DIM, AS, BS);
    prtResult(c_GPU_simple, AS, BS);
    printf("GPU simple method running time: %.2f us\n", elapsedTime_simple * 1e3);

    // ===> CUDA, Tilling (tile width = 2)
    float c_GPU_tilling_2[AS * BS];
    float elapsedTime_tilling_2 = RBF_CUDA_Tilling_2(sigma, a, b, c_GPU_tilling_2, DIM, AS, BS);
    prtResult(c_GPU_tilling_2, AS, BS);
    printf("GPU tilling method (width 2) running time: %.2f us\n", elapsedTime_tilling_2 * 1e3);

    // ===> CUDA, Tilling (tile width = 4)
    float c_GPU_tilling_4[AS * BS];
    float elapsedTime_tilling_4 = RBF_CUDA_Tilling_4(sigma, a, b, c_GPU_tilling_4, DIM, AS, BS);
    prtResult(c_GPU_tilling_4, AS, BS);
    printf("GPU tilling method (width 4) running time: %.2f us\n", elapsedTime_tilling_4 * 1e3);

    // ===> CUDA, Tilling (tile width = 8)
    float c_GPU_tilling_8[AS * BS];
    float elapsedTime_tilling_8 = RBF_CUDA_Tilling_8(sigma, a, b, c_GPU_tilling_8, DIM, AS, BS);
    prtResult(c_GPU_tilling_8, AS, BS);
    printf("GPU tilling method (width 8) running time: %.2f us\n", elapsedTime_tilling_8 * 1e3);

    // ===> CUDA, Tilling (tile width = 16)
    float c_GPU_tilling_16[AS * BS];
    float elapsedTime_tilling_16 = RBF_CUDA_Tilling_16(sigma, a, b, c_GPU_tilling_16, DIM, AS, BS);
    prtResult(c_GPU_tilling_16, AS, BS);
    printf("GPU tilling method (width 16) running time: %.2f us\n", elapsedTime_tilling_16 * 1e3);

    // ===> CUDA, Tilling (tile width = 32)
    float c_GPU_tilling_32[AS * BS];
    float elapsedTime_tilling_32 = RBF_CUDA_Tilling_32(sigma, a, b, c_GPU_tilling_32, DIM, AS, BS);
    prtResult(c_GPU_tilling_32, AS, BS);
    printf("GPU tilling method (width 32) running time: %.2f us\n", elapsedTime_tilling_32 * 1e3);

    return 0;
}
