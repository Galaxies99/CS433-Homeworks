# include <omp.h>
# include <time.h>
# include <vector>
# include <stdio.h>
# include <stdlib.h>
# include <iostream>
# include <sys/time.h>

using namespace std;

const double epsilon = 1e-6;

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
    vector <double> vec;
    srand(time(0));
    for (int i = 0; i < 100000000; ++ i) vec.push_back(i + rand() % 50000000);
    # endif

    double duration, standard_value;
    double res;
    bool correct = true;

    timeval start, end;
    gettimeofday(&start, 0);
    res = sum(vec, false);
    standard_value = res;
    gettimeofday(&end, 0);
    duration = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

    cout << "The sum of the vector: " << res << ", total time usage: " << duration << " second(s) (without OpenMP, warmup).\n";

    gettimeofday(&start, 0);
    res = sum(vec, false);
    gettimeofday(&end, 0);
    correct &= (abs(res - standard_value) < epsilon);
    duration = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

    cout << "The sum of the vector: " << res << ", total time usage: " << duration << " second(s) (without OpenMP).\n";
    
    for (int thread = 1; thread <= 16; thread *= 2) {
        gettimeofday(&start, 0);
        res = sum(vec, true, thread);
        gettimeofday(&end, 0);
        correct &= (abs(res - standard_value) < epsilon);
        duration = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
        cout << "The sum of the vector: " << res << ", total time usage: " << duration << " second(s) (with OpenMP, " << thread << " thread(s)).\n";
    }   

    cout << "Value Check: OpenMP is " << (correct ? "correct" : "incorrect") << ".\n";

    return 0;
}