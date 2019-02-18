#include "distTiling.h"

using namespace std;

int distExec(PriorityQ<rect_t> rQueue, cv::Mat& inImg) {

    int np, rank;

    // initialize mpi and separate manager from worker nodes
    if (MPI_Init(&argc, &argv) != 0) {
        cout << "Init error" << endl;
        exit(1);
    }

    if (MPI_Comm_size(MPI_COMM_WORLD, &np) != 0 
            || MPI_Comm_rank(MPI_COMM_WORLD, &rank) != 0) {
        cout << "Rank error" << endl;
        exit(1);
    }

    // node 0 is the manager
    if (rank == 0) {
        // create a send/recv thread for each 
        for (int p=1; p<=np; p++) {
            sendRecvThread();
        }

        // wait for all threads
    } else {
        // receive either a new tile or an empty tag, signaling the end
        cv::Mat curTile;
        cv::Mat outTile = cv::Mat::zeros(curTile.size(), curTile.type());
        
        // keep processing tiles while there are tiles
        // recvTile returns the tile size, returning 0 if the
        //   tile is empty, signaling that there are no more tiles
        while (recvTile(curTile) > 0) {
            // allocate halide buffers
            h_curTile = Halide::Buffer<uint8_t>(curTile.data, curTile.cols, curTile.rows);
            h_outTile = Halide::Buffer<uint8_t>(outTile.data, outTile.cols, outTile.rows);

            // execute blur 
            blurAOT(h_curTile, h_outTile);
            
            // return executed tile
            sendTile(outTile);
        }
        
    }

    MPI_Finalize();
    return 0;
}

int serializeMat(cv::Mat& m, char *buffer) {
    // size = rows * columns * channels * sizeof type
    int size = m.total() * m.elemSize();

    // add number of rows, columns, channels and data type size
    size += 4*sizeof(int);

    // create the buffer
    matBuffer = (char*) malloc(size);

    // add basic fields of a mat
    memcpy(&buffer[0*sizeof(int)], (char*)&m.rows(), sizeof(int));
    memcpy(&buffer[1*sizeof(int)], (char*)&m.cols(), sizeof(int));
    memcpy(&buffer[2*sizeof(int)], (char*)&m.type(), sizeof(int));
    memcpy(&buffer[3*sizeof(int)], (char*)&m.channels(), sizeof(int));

    // mat data need to be contiguous in order to use memcpy
    if(!m.isContinuous()){ 
        m = m.clone();
    }

    // copy actual data to buffer
    memcpy(&buffer[4*sizeof(int)], m.data, m.total()*m.channels());

    return size;
}

void* sendRecvThread(void *args) {
    rect_t r;
    int currentRank;
    cv::Mat input;

    // keep getting new rect's from the queue
    while (rQueue.pop(r) != 0) {
        // generate a submat from the rect's queue
        cv::Mat subm = input.submat(new cv::Rect(r.xi, r.yi, r.xo, r.yo));
        
        // serialize the mat
        char* buffer;
        int bufSize = serializeMat(subm, buffer);
        
        // send the tile size and the tile itself
        MPI_send(&matSize, 1, MPI_INT, currentRank, MPI_TAG, MPI_COMM_WORLD);
        MPI_Send(&buffer, bufSize, MPI_UNSIGNED_CHAR, 
            currentRank, MPI_TAG, MPI_COMM_WORLD);

        // wait for the resulting mat
        // obs: the same sent buffer is used for receiving since they must
        //   have the same size.
        MPI_Recv(&buffer, bufSize, MPI_UNSIGNED_CHAR, 
            currentRank, MPI_TAG, MPI_COMM_WORLD, NULL);

        // make a mat object from the returned buffer data
        cv::Mat resultm;
        deserializeMat(resultm, buffer);

        // copy result data back to input
        // NOT THREAD-SAFE (ok because there is no overlap)
        Mat aux = input.colRange(r.yi, r.yo).rowRange(r.xi, r.xo);
        resultm.copyTo(aux);

        free(buffer);
    }   

}