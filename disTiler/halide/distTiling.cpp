#include "distTiling.h"

using namespace std;

int serializeMat(cv::Mat& m, char *buffer) {
    // size = rows * columns * channels * sizeof type
    int size = m.total() * m.elemSize();

    // add number of rows, columns, channels and data type size
    size += 4*sizeof(int);

    // create the buffer
    buffer = new char(size);

    // add basic fields of a mat
    int rows = m.rows;
    int cols = m.cols;
    int type = m.type();
    int channels = m.channels();
    memcpy(&buffer[0*sizeof(int)], (char*)&rows, sizeof(int));
    memcpy(&buffer[1*sizeof(int)], (char*)&cols, sizeof(int));
    memcpy(&buffer[2*sizeof(int)], (char*)&type, sizeof(int));
    memcpy(&buffer[3*sizeof(int)], (char*)&channels, sizeof(int));

    // mat data need to be contiguous in order to use memcpy
    if(!m.isContinuous()){ 
        m = m.clone();
    }

    // copy actual data to buffer
    memcpy(&buffer[4*sizeof(int)], m.data, m.total()*m.channels());

    return size;
}

void deserializeMat(cv::Mat& m, char* buffer) {
    // get the basic fields of the mat
    int rows, cols, type, channels;
    memcpy((char*)&rows, &buffer[0*sizeof(int)], sizeof(int));
    memcpy((char*)&cols, &buffer[1*sizeof(int)], sizeof(int));
    memcpy((char*)&type, &buffer[2*sizeof(int)], sizeof(int));
    memcpy((char*)&channels, &buffer[3*sizeof(int)], sizeof(int));//NOT USING YET?

    // create the new mat with the basic fields and the actual data
    m = cv::Mat(rows, cols, type, &buffer[4*sizeof(int)]);
}

// pops the first element of a list, returning 0 if the list is empty
template <typename T>
int pop(const std::list<T>* l, T& out) {
    if (l->size() == 0)
        return 0;
    else {
        out = *(l->begin());
        return 1;
    }
}

void* sendRecvThread(void *args) {
    rect_t r;
    int currentRank = ((thr_args_t*)args)->currentRank;
    cv::Mat* input = ((thr_args_t*)args)->input;
    std::list<rect_t>* rQueue = ((thr_args_t*)args)->rQueue;

    std::cout << "[sendRecvThread] Manager thread " 
            << currentRank << " started" << std::endl;

    // keep getting new rect's from the queue
    while (pop(rQueue, r) != 0) {
        // std::cout << r.xi << "," << r.yi << "," 
        //     << r.xo << "," << r.yo << std::endl;
        // std::cout << r.yo-r.yi << "|" << r.xo-r.xi << std::endl;
        // std::cout << "size: " << input->cols << "," 
        //     << input->rows << std::endl;
        
        // generate a submat from the rect's queue
        cv::Mat subm = input->colRange(r.yi, r.yo).rowRange(r.xi, r.xo);
        
        // serialize the mat
        char* buffer;
        int bufSize = serializeMat(subm, buffer);

        // send the tile size and the tile itself
        MPI_Send(&bufSize, 1, MPI_INT, currentRank, MPI_TAG, MPI_COMM_WORLD);
        MPI_Send(&buffer, bufSize, MPI_UNSIGNED_CHAR, 
            currentRank, MPI_TAG, MPI_COMM_WORLD);

        std::cout << "4" << std::endl;

        // wait for the resulting mat
        // obs: the same sent buffer is used for receiving since they must
        //   have the same size.
        MPI_Recv(&buffer, bufSize, MPI_UNSIGNED_CHAR, 
            currentRank, MPI_TAG, MPI_COMM_WORLD, NULL);

        std::cout << "5" << std::endl;

        // make a mat object from the returned buffer data
        cv::Mat resultm;
        deserializeMat(resultm, buffer);

        std::cout << "6" << std::endl;

        // copy result data back to input
        // NOT THREAD-SAFE (ok because there is no overlap)
        cv::Mat aux = input->colRange(r.yi, r.yo-r.yi).rowRange(r.xi, r.xo-r.xi);
        resultm.copyTo(aux);

        std::cout << "7" << std::endl;

        free(buffer);
    }

    std::cout << "8" << std::endl;

    // send final empty message, signaling the end
    int end = 0;
    MPI_Send(&end, 1, MPI_INT, currentRank, MPI_TAG, MPI_COMM_WORLD);

    std::cout << "9" << std::endl;

    return NULL;
}

int recvTile(cv::Mat& tile, int rank) {
    
    // get the mat object size
    std::cout << "[recvTile][" << rank << "] Waiting " << std::endl;
    int bufSize = 0;
    MPI_Recv(&bufSize, 1, MPI_INT, MPI_MANAGER_RANK, 
        MPI_TAG, MPI_COMM_WORLD, NULL);

    std::cout << "[recvTile][" << rank << "] received size " 
        << bufSize << std::endl;

    // get the actual data, if there is any
    if (bufSize > 0) {
        char* buffer = new char[bufSize];

        MPI_Recv(buffer, bufSize, MPI_UNSIGNED_CHAR, 
            MPI_MANAGER_RANK, MPI_TAG, MPI_COMM_WORLD, NULL);
        std::cout << "[recvTile][" << rank << "] received tile " << std::endl;

        // generate the output mat from the received data
        deserializeMat(tile, buffer);
        delete[] buffer;
        std::cout << "[recvTile][" << rank << "] tile deserialized" << std::endl;
    }

    return bufSize;
}

void sendTile(cv::Mat& tile) {

}

int distExec(int argc, char* argv[], cv::Mat& inImg, cv::Mat& outImg) {

    int np, rank;

    // initialize mpi and separate manager from worker nodes
    if (MPI_Init(&argc, &argv) != 0) {
        cout << "[distExec] Init error" << endl;
        exit(1);
    }

    if (MPI_Comm_size(MPI_COMM_WORLD, &np) != 0 
            || MPI_Comm_rank(MPI_COMM_WORLD, &rank) != 0) {
        cout << "[distExec] Rank error" << endl;
        exit(1);
    }

    // node 0 is the manager
    if (rank == 0) {
        std::list<rect_t> rQueue = autoTiler(inImg);
        std::cout << "[distExec] Starting manager with " 
            << rQueue.size() << " tiles" << std::endl;

        // create a send/recv thread for each worker
        pthread_t threadsId[np-1];
        for (int p=1; p<np; p++) {
            thr_args_t* args = new thr_args_t();
            args->currentRank = p;
            args->input = &inImg;
            args->rQueue = &rQueue;
            pthread_create(&threadsId[p-1], NULL, sendRecvThread, args);
        }

        std::cout << "[distExec] Manager waiting for comm threads to finish\n";

        // wait for all threads to finish
        for (int p=1; p<=np; p++) {
            pthread_join(threadsId[p-1], NULL);
        }
    } else {
        std::cout << "[distExec] Starting worker " << rank << std::endl;

        // receive either a new tile or an empty tag, signaling the end
        cv::Mat curTile;
        cv::Mat outTile = cv::Mat::zeros(curTile.size(), curTile.type()); // ???

        // keep processing tiles while there are tiles
        // recvTile returns the tile size, returning 0 if the
        //   tile is empty, signaling that there are no more tiles
        int bufSize = 0;
        std::cout << "[distExec][" << rank << "] Waiting new tile" << std::endl;
        while ((bufSize = recvTile(curTile, rank)) > 0) {
            std::cout << "[distExec][" << rank << "] Got tile" << std::endl;

            // allocate halide buffers
            Halide::Runtime::Buffer<uint8_t> h_curTile = 
                Halide::Runtime::Buffer<uint8_t>(
                curTile.data, curTile.cols, curTile.rows);
            Halide::Runtime::Buffer<uint8_t> h_outTile = 
                Halide::Runtime::Buffer<uint8_t>(
                outTile.data, outTile.cols, outTile.rows);

            // execute blur 
            std::cout << "[distExec][" << rank << "] Executing" << std::endl;
            blurAOT(h_curTile, h_outTile);
            
            // serialize the output tile
            char* buffer;
            if (bufSize != serializeMat(outTile, buffer)) {
                std::cout << "[distExec] Output tile have a different " 
                    << "size from the input one." << std::endl;
                exit(1);
            }

            // send it back to the manager
            std::cout << "[distExec][" << rank 
                << "] Sending result" << std::endl;
            MPI_Send(&buffer, bufSize, MPI_UNSIGNED_CHAR, 
                rank, MPI_TAG, MPI_COMM_WORLD);
            std::cout << "[distExec][" << rank << "] Waiting new tile" 
                << std::endl;
        }
        std::cout << "[distExec][" << rank << "] Finished" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
