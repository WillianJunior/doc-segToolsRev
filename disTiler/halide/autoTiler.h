#ifndef AUTO_TILER_H
#define AUTO_TILER_H

#define CV_MAX_PIX_VAL 255
#define CV_THR_BIN 0

#include <list>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "distTiling.h"

std::list<rect_t> autoTiler(cv::Mat& input, int border=10, 
    int bgThreshold=50, int erosionSize=10);

#endif
