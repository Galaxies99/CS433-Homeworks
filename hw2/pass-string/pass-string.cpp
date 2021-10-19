/*
Pass-string problem using OpenMPI.

Author: Hongjie Fang
Date: Oct 19th, 2021.
*/
# include <mpi.h>
# include <stdio.h>
# include <string.h>
# include <iostream>

using namespace std;

int main() {
    ios :: sync_with_stdio(false);
    cin.tie(0); cout.tie(0);

    int my_rank, comm_sz; // the rank of current process, and the total number of processes.
    // Initialize MPI processes.
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    // The total number of process should be greater than 2.
    if (comm_sz < 2) {
        cout << "[Error] Process number must be no less than 2.\n";
        return 0;
    }

    string passage; // the initial passage / sentence.
    char *psg; // the passage stored in character array.
    int length; // the length of the passage.

    if (my_rank == 0) {
        // Process 0: (child 0) generate the passage and pass the passage to the 1st child.
        /*
        If define AUTO_EXAMPLE (L40), the source code will generate a default passage;
        otherwise, you may enter the passage yourselves.
        */
        # define AUTO_EXAMPLE
        # ifdef AUTO_EXAMPLE
        passage = "Hello world!";
        # else
        getline(cin, passage);
        # endif
        
        length = passage.size();
        psg = new char[length];
        for (int i = 0; i < length; ++ i) 
            psg[i] = passage[i];
        
        // Pass (send) the passage to the 1st process (child) with tag 0.
        MPI_Send(psg, length, MPI_CHAR, 1, 0, MPI_COMM_WORLD);

        // De-allocate the spaces.
        delete [] psg;
    } else {
        // Process k (k > 0): (child k) receive the passage from the previous child, write down the first word and pass the remaining words to next child.
        bool finish;
        int flag = 0, flag_finish = 0;
        MPI_Status status;
        
        // Check the message repeatedly.
        while (true) {
            // If get the message with tag 0 from the previous process (child).
            MPI_Iprobe(my_rank - 1, 0, MPI_COMM_WORLD, &flag, &status);

            if (flag == 1) {
                // Get the length of the message.
                MPI_Get_count(&status, MPI_CHAR, &length);

                if(length == 0) {
                    // Empty message means that the passage is over. Then the first child with no word received write down his ID.
                    finish = true;
                    cout << "Process " << my_rank << ": " << my_rank << endl;

                    // Pass the finish message to the following child with tag 1.
                    if (my_rank + 1 < comm_sz)
                        MPI_Send(&finish, 1, MPI_CXX_BOOL, my_rank + 1, 1, MPI_COMM_WORLD);
                } else {
                    // Allocate spaces for message.
                    psg = new char[length];

                    // Receive the message from the previous process (child).
                    MPI_Recv(psg, length, MPI_CHAR, my_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    // Get the next word.
                    int nxt = 0;
                    while(nxt < length && psg[nxt] != ' ' && psg[nxt] != '\t' && psg[nxt] != '\n') ++ nxt;

                    // Print the word in the screen.
                    cout << "Process " << my_rank << ": ";
                    for (int i = 0; i < nxt; ++ i) cout << psg[i];
                    cout << endl;

                    // Eliminate duplicate spaces, tab and blank lines.
                    while(nxt < length && (psg[nxt] == ' ' || psg[nxt] == '\t' || psg[nxt] == '\n')) ++ nxt;

                    // Pass (send) the rest passage to the next process (child) with tag 0.
                    if (my_rank + 1 < comm_sz) 
                        MPI_Send(psg + nxt, length - nxt, MPI_CHAR, my_rank + 1, 0, MPI_COMM_WORLD);
                    
                    // De-allocate the spaces.
                    delete [] psg;
                }
                break;
            }
            // If get the message with tag 1 from the previous process (child), that means the string passing is finished.
            MPI_Iprobe(my_rank - 1, 1, MPI_COMM_WORLD, &flag_finish, &status);

            // Pass the finish message to the following process.
            if (flag_finish == 1) {
                // Receive the message from the previous process (child).
                MPI_Recv(&finish, 1, MPI_CXX_BOOL, my_rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Value check.
                if (! finish) cout << "[Error] Unexpected error occured!\n";

                // Pass (send) the finish message to the next process (child) with tag 1.
                if (my_rank + 1 < comm_sz)
                    MPI_Send(&finish, 1, MPI_CXX_BOOL, my_rank + 1, 1, MPI_COMM_WORLD);
                break;
            }
        }
    }
    // Finalize MPI processes.
    MPI_Finalize();
    return 0;
}