#ifndef POPS_H_
#define POPS_H_

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "ConnComponents.hpp"

#define UNVISITED 0x808080
#define BORDER    0x818181
#define MASK      0x80000000
#define RIDGE     0

namespace serial {

template <typename T>
cv::Mat invert(const cv::Mat &img);

template <typename T>
inline void propagate(const cv::Mat& image, cv::Mat& output, std::queue<int>& xQ, 
    std::queue<int>& yQ, int x, int y, T* iPtr, T* oPtr, const T& pval);

template <typename T>
cv::Mat imreconstruct(const cv::Mat& seeds, const cv::Mat& image, int connectivity);

// inclusive min, exclusive max
cv::Mat bwareaopen2(const cv::Mat& image, bool labeled, bool flatten, int minSize, 
    int maxSize, int connectivity, int& count);

template <typename T>
inline void propagateBinary(const cv::Mat& image, cv::Mat& output, std::queue<int>& xQ, 
    std::queue<int>& yQ, int x, int y, T* iPtr, T* oPtr, const T& foreground);

/** optimized serial implementation for binary,
 from Vincent paper on "Morphological Grayscale Reconstruction in Image 
 Analysis: Applicaitons and Efficient Algorithms" connectivity is either 
 4 or 8, default 4.  background is assume to be 0, foreground is assumed 
 to be NOT 0.
 */
template <typename T>
cv::Mat imreconstructBinary(const cv::Mat& seeds, const cv::Mat& image, int connectivity);

// Operates on BINARY IMAGES ONLY
template <typename T>
cv::Mat bwselect(const cv::Mat& binaryImage, const cv::Mat& seeds, int connectivity);

template <typename T>
cv::Mat imfillHoles(const cv::Mat& image, bool binary, int connectivity);

template <typename T>
cv::Mat imhmin(const cv::Mat& image, T h, int connectivity);

// only works with integer images
template <typename T>
cv::Mat_<unsigned char> localMaxima(const cv::Mat& image, int connectivity);

template <typename T>
cv::Mat_<unsigned char> localMinima(const cv::Mat& image, int connectivity);

// Operates on BINARY IMAGES ONLY
// perform bwlabel using union find.
cv::Mat_<int> bwlabel2(const cv::Mat& binaryImage, int connectivity, bool relab);

// require padded image.
template <typename T>
cv::Mat border(cv::Mat& img, T background);

// input should have foreground > 0, and 0 for background
cv::Mat_<int> watershed(const cv::Mat& image, int connectivity);

// input should have foreground > 0, and 0 for background
cv::Mat_<int> watershed2(const cv::Mat& origImage, 
    const cv::Mat_<float>& image, int connectivity);

template <typename T>
cv::Mat morphOpen(const cv::Mat& image, const cv::Mat& kernel);

}

#endif