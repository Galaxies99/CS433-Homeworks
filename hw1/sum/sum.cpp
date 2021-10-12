/*
Calculate the summation of a vector using OpenMP.

Author: Hongjie Fang
Date: Oct 12th, 2021.
*/
# include <omp.h>
# include <time.h>
# include <vector>
# include <stdio.h>
# include <stdlib.h>
# include <iostream>
# include <sys/time.h>

using namespace std;

const double epsilon = 1e-6;

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

int main() {
    ios :: sync_with_stdio(false);
    cin.tie(0); cout.tie(0);

    /*
    If define AUTO_GENERATION (L47), the source code will generate a vector automatically;
    otherwise, you may enter the vector yourselves.
    */
    # define AUTO_GENERATION
    # ifndef AUTO_GENERATION
    int n;
    cout << "Please input the length of the vector: ";
    cin >> n;

    vector <int> vec;
    cout << "Please input the element of the vector, seperated by spaces.\n";
    for (int i = 1; i <= n; ++ i) {
        int temp;
        cin >> temp;
        vec.push_back(temp);
    }
    # else
    vector <double> vec;
    srand(time(0));
    for (int i = 0; i < 100000000; ++ i) vec.push_back(i + rand() % 50000000);
    # endif

    double res; // the result of summation.
    double duration, standard_value; // execution time; the standard value calculated by the normal summation.
    bool correct = true; // whether the result of summation using OpenMP matches with result of normal summation.
    timeval start, end; // since clock() is not accurate, we use gettimeofday(...) in 'sys/time.h' to calculate running time.

    // Normal summation (warmup).
    gettimeofday(&start, 0);
    standard_value = sum(vec, false);
    gettimeofday(&end, 0);
    duration = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    cout << "The sum of the vector: " << res << ", total time usage: " << duration << " second(s) (without OpenMP, warmup).\n";

    // Normal summation.
    // Why do we need a warmup summation? We can make full use of cache after a warmup summation for fair comparisons.
    gettimeofday(&start, 0);
    res = sum(vec, false);
    gettimeofday(&end, 0);
    correct &= (abs(res - standard_value) < epsilon);
    duration = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    cout << "The sum of the vector: " << res << ", total time usage: " << duration << " second(s) (without OpenMP).\n";
    
    // OpenMP summation using different number of threads.
    for (int thread = 1; thread <= 16; thread += 1) {
        gettimeofday(&start, 0);
        res = sum(vec, true, thread);
        gettimeofday(&end, 0);
        correct &= (abs(res - standard_value) < epsilon);
        duration = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
        cout << "The sum of the vector: " << res << ", total time usage: " << duration << " second(s) (with OpenMP, " << thread << " thread(s)).\n";
    }   

    // Correctness check: whether OpenMP results match normal summation result.
    cout << "Value Check: OpenMP is " << (correct ? "correct" : "incorrect") << ".\n";

    return 0;
}