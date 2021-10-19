# CS433 Multicore Architecture and Parallel Programming - Assignment on OpenMP Programming

Hongjie Fang, 518030910150, [galaxies@sjtu.edu.cn](mailto:galaxies@sjtu.edu.cn)

## Environment Setup

Experiment environment is:

- macOS Big Sur 11.5
- 2 GHz Quad-Core Intel Core i5
- 16 GB 3733MHz LPDDR4X

**Note.** Since `makefile` is written under macOS, which has a different compile option with other OS when compiling with OpenMP, we only guarantee that `makefile` can work under the previous system settings (macOS Big Sur 11.5).

## Vector Summation

**Problem Statement**. Please write a program in OpenMP, to compute the sum of a vector.

**Solution**. The core of my implementation is shown as follows.

```cpp
/*
Calculate the summation of a vector.

Params:
    - vec: vector <T>, required, the given vector;
    - with_openmp: bool, optional, default: true, whether to use OpenMP;
    - thread_count: int, optional, default: 4, the number of threads if use OpenMP.

Returns:
    - res: type T, the summation of the given vector.
*/
template <typename T>
T sum(const vector <T> &vec, bool with_openmp = true, int thread_count = 4) {
    T res = 0;
    int i, size = vec.size();
    if (with_openmp) {
        # pragma omp parallel for num_threads(thread_count) reduction(+: res) private(i)
        for (i = 0; i < size; ++ i)
            res += vec[i];
    } else {
        for (i = 0; i < size; ++ i)
            res += vec[i];
    }
    return res;
}
```

Here are the experiment results. We randomly generate a vector of length 100,000,000 for testing. Notice that here we use a warmup normal summation to put the element in the cache for fair comparisons.

| Method | Execution Time (sec) | Speedup Ratio |
| --- | --- | --- |
| Normal Summation (warmup) | 0.303330 | 1.000 |
| Normal Summation | 0.279826 | 1.084 |
| OpenMP (1 thread) | 0.292134 | 1.038 |
| OpenMP (2 thread) | 0.157381 | 1.927 |
| OpenMP (3 thread) | 0.104507 | 2.902 |
| OpenMP (4 thread) | 0.090653 | 3.346 |
| OpenMP (5 thread) | **0.088246** | **3.437** |
| OpenMP (6 thread) | 0.090526 | 3.351 |
| OpenMP (7 thread) | 0.088712 | 3.419 |
| OpenMP (8 thread) | 0.089946 | 3.372 |
| OpenMP (9 thread) | 0.097139 | 3.123 |
| OpenMP (10 thread) | 0.096117 | 3.156 |
| OpenMP (11 thread) | 0.095005 | 3.193 |
| OpenMP (12 thread) | 0.096970 | 3.128 |
| OpenMP (13 thread) | 0.095901 | 3.163 |
| OpenMP (14 thread) | 0.093658 | 3.239 |
| OpenMP (15 thread) | 0.090558 | 3.350 |
| OpenMP (16 thread) | 0.096819 | 3.133 |

From the experiment results we can see that

1. OpenMP can accelerate vector summation up to a speedup ratio of 3.437, which shows the power of parallel computing.
2. OpenMP has additional cost (See comparison between OpenMP (1 thread) and normal summation).

## Matrix Multiplication

**Problem Statement**. Please implement a function to compute matrix multiplication in OpenMP.
**Solution.** The core of my implementation is shown as follows. I implement three types of optimization, namely

- (Method 1) normal OpenMP optimization (parallel for);
- (Method 2) normal OpenMP optimization (parallel for), along with transposed matrix to improve locality;
- (Method 3) normal OpenMP optimization (parallel for), along with block matrix multiplication to improve locality.

```cpp
/*
Calculate the matrix multiplication.

Params:
    - matA, matB: 2-dimension array of size T, the given matrices;
    - matC: 2-dimension array of size T, the result matrix;
    - M, N, K: the size of matrices, matA should be of size M x N, matB should be of size N x K;
    - with_openmp: bool, optional, default: true, whether to use OpenMP;
    - thread_count: int, optional, default: 4, the number of threads if use OpenMP.
    - optim_type: non-negative int, default: 0, the optimization type if use OpenMP.
        + optim_type = 0: normal OpenMP;
        + optim_type = 1: normal OpenMP with transposed matrix matB;
        + optim_type >= 2: matrix multiplication using blocks (block size = optim_type).

Returns:
    - the return code of the function (0: successful, otherwise: error occurred).
*/
template <typename T>
int MatrixMultiplication(T **matA, T **matB, T **matC, int M, int N, int K, bool with_openmp = true, int thread_count = 4, int optim_type = 0) {
    int i, j, k;
    if (with_openmp) {
        // Check type of optimization.
        if (optim_type < 0) return -1;
        else if (optim_type == 0) { // normal OpenMP
            # pragma omp parallel for num_threads(thread_count) private(i)
            for (i = 0; i < M; ++ i)
                # pragma omp parallel for num_threads(thread_count) private(j)
                for (j = 0; j < K; ++ j) {
                    T tmp_sum = 0;
                    # pragma omp parallel for num_threads(thread_count) reduction(+: tmp_sum) private(k)
                    for (k = 0; k < N; ++ k)
                        tmp_sum += matA[i][k] * matB[k][j];
                    matC[i][j] = tmp_sum;
                }
            return 0;
        } else if (optim_type == 1) { // normal OpenMP with transposed matrix matB;
            T **matB_tr = new T*[K];
            # pragma omp parallel for num_threads(thread_count) private(i)
            for (i = 0; i < K; ++ i)
                matB_tr[i] = new T[N];
            # pragma omp parallel for num_threads(thread_count) private(i)
            for (i = 0; i < N; ++ i)
                # pragma omp parallel for num_threads(thread_count) private(j)
                for (j = 0; j < K; ++ j)
                    matB_tr[j][i] = matB[i][j];
            # pragma omp parallel for num_threads(thread_count) private(i)
            for (i = 0; i < M; ++ i)
                # pragma omp parallel for num_threads(thread_count) private(j)
                for (j = 0; j < N; ++ j) {
                    T tmp_sum = 0;
                    # pragma omp parallel for num_threads(thread_count) reduction(+: tmp_sum) private(k)
                    for (k = 0; k < N; ++ k)
                        tmp_sum += matA[i][k] * matB_tr[j][k];
                    matC[i][j] = tmp_sum;
                }
            # pragma omp parallel for num_threads(thread_count) private(i)
            for (i = 0; i < K; ++ i)
                delete [] matB_tr[i];
            delete [] matB_tr;
            return 0;
        } else { // optim_type >= 2: matrix multiplication using blocks (block size = optim_type).
            int block_size = optim_type;
            # pragma omp parallel for num_threads(thread_count) private(i)
            for (i = 0; i < N; ++ i)
                # pragma omp parallel for num_threads(thread_count) private(j)
                for (j = 0; j < K; ++ j)
                    matC[i][j] = 0;
            for (i = 0; i < M; i += block_size) {
                int iito = min(i + block_size, M);
                for (j = 0; j < K; j += block_size) {
                    int jjto = min(j + block_size, K);
                    for (k = 0; k < N; k += block_size) {
                        int ii, jj, kk, kkto = min(k + block_size, N);
                        # pragma omp parallel for num_threads(thread_count) private(ii)
                        for (ii = i; ii < iito; ++ ii)
                            # pragma omp parallel for num_threads(thread_count) private(jj)
                            for (jj = j; jj < jjto; ++ jj)
                                # pragma omp parallel for num_threads(thread_count) private(kk)
                                for (kk = k; kk < kkto; ++ kk) {
                                    matC[ii][jj] += matA[ii][kk] * matB[kk][jj];
                                }
                    }
                }
            }
            return 0;
        }
        return -1;
    } else {
        for (i = 0; i < M; ++ i)
            for (j = 0; j < K; ++ j) {
                T tmp_sum = 0;
                for (k = 0; k < N; ++ k)
                    tmp_sum += matA[i][k] * matB[k][j];
                matC[i][j] = tmp_sum;
            }
        return 0;
    }
}
```

Here are the experiment results. We randomly generate two matrices of size 1,000 x 1,000 for testing. Notice that here we use a warmup normal matrix multiplication to put the element in the cache for fair comparisons.

| Method | Execution Time (sec) | Speedup Ratio |
| --- | --- | --- |
| Normal Summation (warmup) | 0.453820 | 1.000 |
| Normal Summation | 0.430041 | 1.055 |
| Method 1 (1 thread) | 0.630116 | 0.720 |
| Method 1 (2 thread) | 0.318573 | 1.425 |
| Method 1 (3 thread) | 0.219403 | 2.068 |
| Method 1 (4 thread) | 0.217267 | 2.089 |
| Method 1 (5 thread) | 0.177667 | 2.554 |
| Method 1 (6 thread) | 0.173782 | 2.611 |
| Method 1 (7 thread) | 0.172786 | 2.626 |
| Method 1 (8 thread) | 0.165128 | **2.748** |
| Method 2 (1 thread) | 0.537427 | 0.845 |
| Method 2 (2 thread) | 0.232192 | 1.955 |
| Method 2 (3 thread) | 0.157058 | 2.890 |
| Method 2 (4 thread) | 0.142963 | 3.174 |
| Method 2 (5 thread) | **0.133811** | **3.391** |
| Method 2 (6 thread) | 0.136246 | 3.331 |
| Method 2 (7 thread) | 0.143094 | 3.171 |
| Method 2 (8 thread) | 0.133833 | **3.391** |

For method 3, we set block size to 8, 16, 32, 64, 128 and 256 respectively to test the performance of different block size. The following table shows the execution time under different experiment settings.

| Thread Number | 8 | 16 | 32 | 64 | 128 | 256 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4.51 | 2.50 | 1.54 | 1.16 | 0.92 | 0.92 |
| 2 | 2.41 | 1.29 | 0.77 | 0.54 | 0.45 | 0.43 |
| 3 | 1.88 | 0.97 | 0.58 | 0.38 | 0.31 | 0.29 |
| 4 | 1.49 | 0.67 | 0.41 | 0.32 | 0.24 |  0.23 |
| 5 | 2.11 | 1.01 | 0.59 | 0.39 | 0.32 | 0.26 |
| 6 | 2.19 | 0.94 | 0.51 | 0.34 | 0.28 | 0.26 |
| 7 | 2.24 | 0.92 | 0.47 | 0.30 | 0.25 | 0.23 |
| 8 | 1.52 | 0.72 | 0.40 | 0.39 | 0.23 | **0.21** |

The maximum speedup ratio of method 3 is 2.073.

From the experiment results we can observe that:

1. Normal OpenMP can accelerate matrix multiplication to  a speedup ratio of 2.748, which shows the power of parallel computing.
2. Transposed matrix trick can improve the space locality and accelerate matrix multiplication to a speedup ratio of 3.391, which shows the importance of locality in efficiency.
3. OpenMP has additional cost (See comparison between OpenMP (1 thread) and normal multiplication).
4. Method 3 performs worse than previous methods. Under some settings it is even worse than normal matrix multiplication, which is because the overhead of OpenMP on multiple loop is expensive, which may worsen the performances.
