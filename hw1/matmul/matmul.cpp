/*
Calculate the matrix multiplication using OpenMP.

Author: Hongjie Fang
Date: Oct 12th, 2021.
*/
# include <omp.h>
# include <math.h>
# include <time.h>
# include <vector>
# include <stdio.h>
# include <stdlib.h>
# include <iostream>
# include <sys/time.h>

using namespace std;

const double epsilon = 1e-6;

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

/*
    Allocate spaces for a matrix.
*/
template <typename T>
T** allocate_mat(T **mat, int M, int N) {
    mat = new T*[M];
    for (int i = 0; i < M; ++ i)
        mat[i] = new T[N];
    return mat;
}

/*
    Release spaces for a matrix.
*/
template <typename T>
void release_mat(T **mat, int M, int N) {
    for (int i = 0; i < M; ++ i)
        delete [] mat[i];
    delete [] mat;
}

/*
    Compare whether two matrices are the same.
*/
template <typename T>
bool mat_comp(T **matA, T **matB, int M, int N) {
    double res = 0;
    for (int i = 0; i < M; ++ i)
        for (int j = 0; j < N; ++ j)
            res += abs(matA[i][j] - matB[i][j]);
    return res < epsilon;
}

int main() {
    ios :: sync_with_stdio(false);
    cin.tie(0); cout.tie(0);

    int M, N, K;
    double **matA, **matB, **matC;

    /*
    If define AUTO_GENERATION, the source code will generate a vector automatically;
    otherwise, you may enter the vector yourselves.
    */
    # define AUTO_GENERATION
    # ifndef AUTO_GENERATION
    cout << "Please input the size of the matrices (M, N, K seperated by spaces): ";
    cin >> M >> N >> K;

    matA = allocate_mat(matA, M, N);
    matB = allocate_mat(matB, N, K);
    matC = allocate_mat(matC, M, K);

    cout << "Please input the element of matA (M x N), seperated by spaces (or enter).\n";
    for (int i = 0; i < M; ++ i) 
        for (int j = 0; j < N; ++ j) cin >> matA[i][j];
    
    cout << "Please input the element of matB (N x K), seperated by spaces (or enter).\n";
    for (int i = 0; i < N; ++ i) 
        for (int j = 0; j < K; ++ j) cin >> matB[i][j];
    # else
    M = 500, N = 500, K = 500;
    matA = allocate_mat(matA, M, N);
    matB = allocate_mat(matB, N, K);
    matC = allocate_mat(matC, M, K);
    for (int i = 0; i < M; ++ i) 
        for (int j = 0; j < N; ++ j) matA[i][j] = rand() % 50000;
    for (int i = 0; i < N; ++ i) 
        for (int j = 0; j < K; ++ j) matB[i][j] = rand() % 50000;
    # endif

    double **std_matC; // standard matrix calculated by normal matrix multiplication.
    std_matC = allocate_mat(std_matC, M, K);

    int res; // the return value of matrix multiplication.
    double duration; // execution time; the standard value calculated by the normal summation.
    bool correct = true; // whether the result of summation using OpenMP matches with result of normal summation.
    timeval start, end; // since clock() is not accurate, we use gettimeofday(...) in 'sys/time.h' to calculate running time.

    // Normal summation (warmup).
    gettimeofday(&start, 0);
    res = MatrixMultiplication(matA, matB, matC, M, N, K, false);
    gettimeofday(&end, 0);
    if (res) {
        cout << "[Error] Error occured during execution.\n";
        return 0;
    } else {
        for (int i = 0; i < M; ++ i)
            for (int j = 0; j < N; ++ j)
                std_matC[i][j] = matC[i][j];
    }
    duration = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    cout << "Total time usage: " << duration << " second(s) (without OpenMP, warmup).\n";

    // Normal summation.
    // Why do we need a warmup summation? We can make full use of cache after a warmup summation for fair comparisons.
    gettimeofday(&start, 0);
    res = MatrixMultiplication(matA, matB, matC, M, N, K, false);
    gettimeofday(&end, 0);
    if (res) 
        cout << "[Error] Error occured during execution.\n";
    else
        correct &= mat_comp(matC, std_matC, M, K);
    duration = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    cout << "Total time usage: " << duration << " second(s) (without OpenMP).\n";
    
    // 1. normal OpenMP.
    for (int thread = 1; thread <= 8; thread += 1) {
        gettimeofday(&start, 0);
        res = MatrixMultiplication(matA, matB, matC, M, N, K, true, thread, 0);
        gettimeofday(&end, 0);
        if (res) 
            cout << "[Error] Error occured during execution.\n";
        else
            correct &= mat_comp(matC, std_matC, M, K);
        duration = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
        cout << "Total time usage: " << duration << " second(s) (with normal OpenMP, " << thread << " thread(s)).\n";
    } 

    // 2. normal OpenMP with transposed matrix matB.
    for (int thread = 1; thread <= 8; thread += 1) {
        gettimeofday(&start, 0);
        res = MatrixMultiplication(matA, matB, matC, M, N, K, true, thread, 1);
        gettimeofday(&end, 0);
        if (res) 
            cout << "[Error] Error occured during execution.\n";
        else
            correct &= mat_comp(matC, std_matC, M, K);
        duration = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
        cout << "Total time usage: " << duration << " second(s) (with normal OpenMP and transposed matrix, " << thread << " thread(s)).\n";
    }  

    // 3. OpenMP with block multiplication.
    for (int block_size = 8; block_size <= 256; block_size *= 2) {
        for (int thread = 1; thread <= 8; thread += 1) {
            gettimeofday(&start, 0);
            res = MatrixMultiplication(matA, matB, matC, M, N, K, true, thread, block_size);
            gettimeofday(&end, 0);
            if (res) 
                cout << "[Error] Error occured during execution.\n";
            else
                correct &= mat_comp(matC, std_matC, M, K);
            duration = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
            cout << "Total time usage: " << duration << " second(s) (with OpenMP and block multiplication of block size " << block_size << ", " << thread << " thread(s)).\n";
        }  
    }

    // Correctness check: whether OpenMP results match normal summation result.
    cout << "Value Check: OpenMP is " << (correct ? "correct" : "incorrect") << ".\n";

    release_mat(matA, M, N);
    release_mat(matB, N, K);
    release_mat(matC, M, K);
    release_mat(std_matC, M, K);
    return 0;
}