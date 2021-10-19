/*
Round-trip test using OpenMPI.

Author: Hongjie Fang
Date: Oct 19th, 2021.
*/
# include <mpi.h>
# include <stdio.h>
# include <iostream>
# include <sys/time.h>

using namespace std;

int main() {
    
    int my_rank, comm_sz; // the rank of current process, and the total number of processes.
    // Initialize MPI processes.
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    // The total number of process should be 2.
    if (comm_sz != 2) {
        cout << "[Error] Process number must be 2.\n";
        return 0;
    }

    timeval start, end; // since clock() is not accurate, we use gettimeofday(...) in 'sys/time.h' to calculate running time.
    double duration; // execution time.

    int ntimes = 1000000; // test times for each type of data.
    int send_initialize = 0, recv_initialize; // initialize before testing, to make the execution time more stable.
    int send_int = 1024, recv_int; // test sample for int type.
    long long send_long_long = 2147483648ll * 32768, recv_long_long; // test sample for long long type. 
    double send_double = 233.3, recv_double; // test sample for double type.
    bool send_bool = true, recv_bool; // test sample for bool type.
    char send_char = 'A', recv_char; // test sample for char type.

    bool ACK = true, recv_ACK; // acknowledgement sign.
    
    if (my_rank == 0) {
        // Processor 0
        // Initialization before testing, to make the execution time more stable.
        for (int i = 1; i <= ntimes; ++ i) {
            MPI_Send(&send_initialize, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&recv_initialize, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Value check.
            if (recv_initialize != 0) {
                cout << "[Error] MPI Initialization Error!\n";
            }
        }

        // Test the round-trip time of int.
        duration = 0;
        for (int i = 1; i <= ntimes; ++ i) {
            gettimeofday(&start, 0);
            MPI_Send(&send_int, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&recv_ACK, 1, MPI_CXX_BOOL, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            gettimeofday(&end, 0);
            // Value check.
            if (! recv_ACK) {
                cout << "[Error] Communication error on int-type value!\n";
            }
            duration += (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
        }
        duration /= ntimes;
        cout << "Round-trip time on int: " << duration << " us.\n";
        
        // Test the round-trip time of long long.
        duration = 0;
        for (int i = 1; i <= ntimes; ++ i) {
            gettimeofday(&start, 0);
            MPI_Send(&send_long_long, 1, MPI_LONG_LONG, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&recv_ACK, 1, MPI_CXX_BOOL, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            gettimeofday(&end, 0);
            // Value check.
            if (! recv_ACK) {
                cout << "[Error] Communication error on long-long-type value!\n";
            }
            duration += (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
        }
        duration /= ntimes;
        cout << "Round-trip time on long long: " << duration << " us.\n";

        // Test the round-trip time of double.
        duration = 0;
        for (int i = 1; i <= ntimes; ++ i) {
            gettimeofday(&start, 0);
            MPI_Send(&send_double, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&recv_ACK, 1, MPI_CXX_BOOL, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            gettimeofday(&end, 0);
            // Value check.
            if (! recv_ACK) {
                cout << "[Error] Communication error on double-type value!\n";
            }
            duration += (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
        }
        duration /= ntimes;
        cout << "Round-trip time on double: " << duration << " us.\n";

        // Test the round-trip time of bool.
        duration = 0;
        for (int i = 1; i <= ntimes; ++ i) {
            gettimeofday(&start, 0);
            MPI_Send(&send_bool, 1, MPI_CXX_BOOL, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&recv_ACK, 1, MPI_CXX_BOOL, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            gettimeofday(&end, 0);
            // Value check.
            if (! recv_ACK) {
                cout << "[Error] Communication error on bool-type value!\n";
            }
            duration += (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
        }
        duration /= ntimes;
        cout << "Round-trip time on bool: " << duration << " us.\n";

        // Test the round-trip time of char.
        duration = 0;
        for (int i = 1; i <= ntimes; ++ i) {
            gettimeofday(&start, 0);
            MPI_Send(&send_char, 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&recv_ACK, 1, MPI_CXX_BOOL, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            gettimeofday(&end, 0);
            // Value check.
            if (! recv_ACK) {
                cout << "[Error] Communication error on char-type value!\n";
            }
            duration += (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
        }
        duration /= ntimes;
        cout << "Round-trip time on char: " << duration << " us.\n";
    } else if (my_rank == 1) {
        // Processor 1
        // Initialization.
        for (int i = 1; i <= ntimes; ++ i) {
            MPI_Recv(&recv_initialize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Value check.
            if (recv_initialize != 0) {
                cout << "[Error] MPI Initialization Error!\n";
            }
            MPI_Send(&send_initialize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }

        // Test the round-trip time of int.
        for (int i = 1; i <= ntimes; ++ i) {
            MPI_Recv(&recv_int, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Value check.
            if (recv_int != 1024) {
                cout << "[Error] Communication error on int-type value!\n";
            }
            MPI_Send(&ACK, 1, MPI_CXX_BOOL, 0, 1, MPI_COMM_WORLD);
        }

        // Test the round-trip time of long long.
        for (int i = 1; i <= ntimes; ++ i) {
            MPI_Recv(&recv_long_long, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Value check.
            if (recv_long_long != 2147483648ll * 32768) {
                cout << "[Error] Communication error on long-long-type value!\n";
            }
            MPI_Send(&ACK, 1, MPI_CXX_BOOL, 0, 1, MPI_COMM_WORLD);
        }

        // Test the round-trip time of double.
        for (int i = 1; i <= ntimes; ++ i) {
            MPI_Recv(&recv_double, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Value check.
            if (recv_double != 233.3) {
                cout << "[Error] Communication error on double-type value!\n";
            }
            MPI_Send(&ACK, 1, MPI_CXX_BOOL, 0, 1, MPI_COMM_WORLD);
        }
        
        // Test the round-trip time of bool.
        for (int i = 1; i <= ntimes; ++ i) {
            MPI_Recv(&recv_bool, 1, MPI_CXX_BOOL, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Value check.
            if (recv_bool != true) {
                cout << "[Error] Communication error on bool-type value!\n";
            }
            MPI_Send(&ACK, 1, MPI_CXX_BOOL, 0, 1, MPI_COMM_WORLD);
        }

        // Test the round-trip time of char.
        for (int i = 1; i <= ntimes; ++ i) {
            MPI_Recv(&recv_char, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Value check.
            if (recv_char != 'A') {
                cout << "[Error] Communication error on char-type value!\n";
            }
            MPI_Send(&ACK, 1, MPI_CXX_BOOL, 0, 1, MPI_COMM_WORLD);
        }
    }
    // Finalize MPI processes.
    MPI_Finalize();
    return 0;
}