#ifndef DISTILING_H
#define DISTILING_H 

#include "mpi.h"
#include <pthread.h>
#include <opencv2/opencv.hpp>
#include "Halide.h"

#include "PriorityQ.h"

#define MPI_TAG 0

typedef struct rect_t {
	int xi, yi;
	int xo, yo;
} rect_t;

typedef struct thr_args_t {
	int currentRank;
    cv::Mat *input;
    PriorityQ<rect_t> *rQueue;
} thr_args_t;

int distExec(PriorityQ<rect_t> rQueue, cv::Mat& inImg, cv::Mat& outImg);

#endif