# include <omp.h>
# include <time.h>
# include <vector>
# include <stdio.h>
# include <stdlib.h>
# include <iostream>

using namespace std;


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

    # define AUTO_GENERATION
    # ifndef AUTO_GENERATION
    int n;
    cout << "Please input the length of the vector: ";
    cin >> n;

    vector <int> vec;
    cout << "Please input the element of the vector, seperated by space.\n";
    for (int i = 1; i <= n; ++ i) {
        int temp;
        cin >> temp;
        vec.push_back(temp);
    }
    # else
    vector <long long> vec;
    srand(time(0));
    for (int i = 0; i < 100000000; ++ i) vec.push_back(i + rand());
    # endif

    double start_time, duration;
    long long res;

    start_time = clock();
    res = sum(vec, false);
    duration = clock() - start_time;

    cout << "The sum of the vector: " << res << ", total time usage: " << duration / CLOCKS_PER_SEC << " second(s) (without OpenMP, warmup).\n";

    start_time = clock();
    res = sum(vec, false);
    duration = clock() - start_time;

    cout << "The sum of the vector: " << res << ", total time usage: " << duration / CLOCKS_PER_SEC << " second(s) (without OpenMP).\n";
    
    for (int thread = 1; thread <= 8; thread ++) {
        start_time = clock();
        res = sum(vec, true, thread);
        duration = clock() - start_time;
        cout << "The sum of the vector: " << res << ", total time usage: " << duration / CLOCKS_PER_SEC << " second(s) (with OpenMP, " << thread << " extra thread(s)).\n";
    }
    return 0;
}