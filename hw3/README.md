# CS433 Multicore Architecture and Parallel Programming - Assignment on CUDA Programming

Hongjie Fang, 518030910150, [galaxies@sjtu.edu.cn](mailto:galaxies@sjtu.edu.cn)

## Environment Setup

Experiment environment is:

- Ubuntu 18.04.6 LTS
- NVIDIA GeForce RTX 2080Ti
- CUDA Version: 10.1
- CPU: 48 Processors (Intel(R) Xeon(R) CPU E5-2678 v3 @ 2.50GHz)

## RBF Kernel Calculation and Optimizations

**Problem Statement**. The assignment consists of two subproblems.

- Please write CUDA kernel functions to effectively compute the RBF kernel of two matrices. You may need to write wrapper functions to accomplish the task.
- Have you ever experienced the power of GPU computing? To achieve the maximum performance, there are many tricks based on the processor and memory architecture. I want to show you a trick that you may never see in most CUDA documentations released by NVIDIA. The memory hierarchy of CUDA architecture brings us global memory (including constant memory, texture memory and surface memory), shared memory and registers. You must realize that shared memory should be used if possible to avoid high latency accessing global memory. But shared memory is again far slower than registers. How can we achieve the extreme performance? Please read [reference](https://laurel.datsi.fi.upm.es/_media/proyectos/gopac/volkov10-gtc.pdf) and optimize your program to get more speedup.

**Solution**. We assume that matrix `A` has the size of `AS * dim` and matrix `B` has the size of `BS * dim`. Then, the result of `K(A, B)` should have the size of `AS * BS`. The experiment setup is that `AS = 32, BS = 32` due to the block limits of the GPU, and `dim = 1024`. The elements in the matrices are randomly generated. We also write the RBF kernel function in CPU for correctness check, which is shown as follows.

```cpp
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
```

Then, we can write a simple CUDA kernel code to calculate the RBF kernel shown as follows, which is similar to the matrix multiplication implementations.

```cpp
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
```

However, we can notice that the efficacy of the simple implementation is limited. After reading reference material and review the slides of the teacher in the class, I found that we can use tilling methods to accelerate the calculation. Actually the reference material states that we can perform loop unrolling to optimize matrix multiplication, I observe that loop unrolling is actually equivalent with tilling methods. Also, tilling method can fully use the GPU memory hirarchy since it can make use of the shared memory. Therefore, I implement RBF kernel with tilling methods in the following optimization process.

For tilling method, an important variable is the tilling width (`TILE_WIDTH` in professor's slide). Considering the experiment setup, we set `TILE_WIDTH` to `2`, `4`, `8`, `16`, `32` respectively to investigate the speedup difference of different tilling width. Take `TILE_WIDTH = 2` for instance, we implement the following CUDA kernel code.

```cpp
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
```

After implementations, we execute the code to perform experiments. The outputs are shown below.

```text
The sum of the matrix: 62.12
CPU method running time: 10889.00 us
The sum of the matrix: 62.12
GPU simple method running time: 1226.66 us
The sum of the matrix: 62.12
GPU tilling method (width 2) running time: 352.93 us
The sum of the matrix: 62.12
GPU tilling method (width 4) running time: 151.33 us
The sum of the matrix: 62.12
GPU tilling method (width 8) running time: 47.30 us
The sum of the matrix: 62.12
GPU tilling method (width 16) running time: 74.66 us
The sum of the matrix: 62.12
GPU tilling method (width 32) running time: 465.66 us
```

We use the summation of the answer matrix to verify the correctness of our CUDA implementations. From the results we can see that all implementations are correct. Then, we can summarize the results in the following table.

| Method | Tilling Width | Running Time (us) | Speedup Ratio |
| --- | --- | --- | --- |
| CPU | - | 10889.00 | 1.000 |
| CUDA | - | 1226.66 | 8.877 |
| CUDA + Tilling | 2 | 352.93 | 30.853 |
| CUDA + Tilling | 4 | 151.33 | 71.955 |
| CUDA + Tilling | 8 | **47.30** | **230.211** |
| CUDA + Tilling | 16 | 74.66 | 145.848 |
| CUDA + Tilling | 32 | 465.66 | 23.384 |

From the experiment results we can see that the simple CUDA program can reach approximately 9x speedup ratio. However, if we use tilling optimization method to make fully use of the shared memory and the parallel capability, we can further accelerate the RBF kernel calculation and runs 230x faster than the CPU implementation. From this homework I realize that, how to write a good CUDA program to achieve the best performances remains an interesting problem and needs to be investigated further.

**User manual**. Use the following command to compile, execute and clean, respectively.

```bash
# compile
make
# run
./rbf
# clean
make clean
```
