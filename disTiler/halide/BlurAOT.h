#ifndef BLUR_AOT_H
#define BLUR_AOT_H

#include "Halide.h"
#include <opencv2/opencv.hpp>

enum ret_t {
    SUCCESS
};

// struct {
//     int xi, yi; // upper left point
//     int xo, yo; // lower right point
// }tile;

class BlurAOTGen : public Halide::Generator<BlurAOTGen> {
public:
    BlurAOTGen(cv::Mat& input, cv::Mat& output);
    ret_t sched();
    ret_t run();
    cv::Mat getResult();

private:
    cv::Mat input;
    cv::Mat output;

    int rows, cols;

    // halide functions
    Halide::Func blurx;
    Halide::Func hf_output;
    Halide::Func clamped;

    Halide::Buffer<uint8_t> hb_input;
    Halide::Buffer<uint8_t> hb_output;

    Halide::Var x, y;
};

class BlurAOTRun {
public:
    BlurAOTGen(cv::Mat& input);
    ret_t run();
    cv::Mat getResult();
    int sendMpi();
    int recvMpi(char* buf);

private:
    
};

#endif
