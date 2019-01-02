/*
 * RedBloodCell.cpp
 *
 *  Created on: Jul 1, 2011
 *      Author: tcpan
 */

#include <iostream>

#include "pixelsOps.hpp"

#include <opencv2/opencv.hpp>

#define CONTINUE 0
#define BACKGROUND 1
#define BACKGROUND_LIKELY 2
#define NO_CANDIDATES_LEFT 3
#define SUCCESS 4
#define INVALID_IMAGE 5


int segmentNuclei(const std::string& in, const std::string& out,
    int &compcount, int *&bbox, unsigned char blue, unsigned char green, 
    unsigned char red, double T1, double T2, unsigned char G1, int minSize, 
    int maxSize, unsigned char G2, int minSizePl, int minSizeSeg, int maxSizeSeg, 
    int fillHolesConnectivity, int reconConnectivity, int watershedConnectivity);

int main() {

    int *bbox = NULL;
    int compcount;

    unsigned char G1 = 80;
    unsigned char G2 = 40;
    int minSize = 2;
    int reconConnectivity = 8;
    
    double T2 = 7.5;
    int maxSize = 1000;
    int minSizePl = 5;
    int minSizeSeg = 2;
    
    unsigned char blue = 240;
    unsigned char green = 240;
    unsigned char red = 240;
    double T1 = 7.5;
    int maxSizeSeg = 1000;
    int fillHolesConnectivity = 8;
    int watershedConnectivity = 8;

    int ret = segmentNuclei("in.tiff", "out.tiff", compcount, bbox, blue, green, 
        red, T1, T2, G1, minSize, maxSize, G2, minSizePl, minSizeSeg, 
        maxSizeSeg, fillHolesConnectivity, reconConnectivity, watershedConnectivity);

    std::cout << "ret: " << ret << std::endl;
    return 0;
}

// cv::Mat HistologicalEntities::getRBC(const cv::Mat& img,  double T1, double T2,
//      ::cciutils::SimpleCSVLogger *logger, ::cciutils::cv::IntermediateResultHandler *iresHandler) {
//  CV_Assert(img.channels() == 3);

//  std::vector<cv::Mat> bgr;
//  split(img, bgr);
//  return getRBC(bgr, T1, T2, logger, iresHandler);
// }

cv::Mat getRBC(const std::vector<cv::Mat>& bgr,  double T1, double T2) {
    CV_Assert(bgr.size() == 3);
    
    std::cout.precision(5);
    cv::Size s = bgr[0].size();
    cv::Mat bd(s, CV_32FC1);
    cv::Mat gd(s, bd.type());
    cv::Mat rd(s, bd.type());

    bgr[0].convertTo(bd, bd.type(), 1.0, FLT_EPSILON);
    bgr[1].convertTo(gd, gd.type(), 1.0, FLT_EPSILON);
    bgr[2].convertTo(rd, rd.type(), 1.0, 0.0);

    cv::Mat imR2G = rd / gd;
    cv::Mat imR2B = (rd / bd) > 1.0;

    cv::Mat bw1 = imR2G > T1;
    cv::Mat bw2 = imR2G > T2;
    cv::Mat rbc;
    if (countNonZero(bw1) > 0) {
        rbc = serial::bwselect<unsigned char>(bw2, bw1, 8) & imR2B;
    } else {
        rbc = cv::Mat::zeros(bw2.size(), bw2.type());
    }

    return rbc;
}

// cv::Mat HistologicalEntities::getBackground(const cv::Mat& img, unsigned char blue, unsigned char green, unsigned char red,
//      ::cciutils::SimpleCSVLogger *logger, ::cciutils::cv::IntermediateResultHandler *iresHandler) {
//  CV_Assert(img.channels() == 3);

//  std::vector<cv::Mat> bgr;
//  split(img, bgr);
//  return getBackground(bgr, blue, green, red, logger, iresHandler);
// }

cv::Mat getBackground(const std::vector<cv::Mat>& bgr, unsigned char blue, 
    unsigned char green, unsigned char red) {

    return (bgr[0] > blue) & (bgr[1] > green) & (bgr[2] > red);
}

// S1
int plFindNucleusCandidates(const cv::Mat& img, cv::Mat& seg_norbc, unsigned char blue, 
    unsigned char green, unsigned char red, double T1, double T2, 
    unsigned char G1, int minSize, int maxSize, unsigned char G2, 
    int fillHolesConnectivity, int reconConnectivity) {

    std::vector<cv::Mat> bgr;
    split(img, bgr);
    
    cv::Mat background = getBackground(bgr, blue, green, red);

    int bgArea = countNonZero(background);
    float ratio = (float)bgArea / (float)(img.size().area());
    
    if (ratio >= 0.99) {
        return BACKGROUND;
    } else if (ratio >= 0.9) {
        return BACKGROUND_LIKELY;
    }

    cv::Mat rbc = getRBC(bgr, T1, T2);
    int rbcPixelCount = countNonZero(rbc);

    cv::Mat rc = serial::invert<unsigned char>(bgr[2]);

    cv::Mat rc_open(rc.size(), rc.type());
    //cv::Mat disk19 = getStructuringElement(MORPH_ELLIPSE, Size(19,19));
    // structuring element is not the same between matlab and opencv.  
    //    using the one from matlab explicitly....
    // (for 4, 6, and 8 connected, they are approximations).
    unsigned char disk19raw[361] = {
            0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
            0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
            0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0};
    std::vector<unsigned char> disk19vec(disk19raw, disk19raw+361);
    cv::Mat disk19(disk19vec);
    disk19 = disk19.reshape(1, 19);
    rc_open = serial::morphOpen<unsigned char>(rc, disk19);

    cv::Mat rc_recon = serial::imreconstruct<unsigned char>(rc_open, rc, reconConnectivity);

    cv::Mat diffIm = rc - rc_recon;
    int rc_openPixelCount = countNonZero(rc_open);

    // it is now a parameter
    cv::Mat diffIm2 = diffIm > G1;

    cv::Mat bw1 = serial::imfillHoles<unsigned char>(diffIm2, true, fillHolesConnectivity);

    int compcount2;

    cv::Mat bw1_t = serial::bwareaopen2(bw1, false, true, minSize, maxSize, 8, compcount2);

    bw1.release();
    if (compcount2 == 0) {
        return NO_CANDIDATES_LEFT;
    }

    // It is now a parameter
    cv::Mat bw2 = diffIm > G2;
    seg_norbc = serial::bwselect<unsigned char>(bw2, bw1_t, 8);
    seg_norbc = seg_norbc & (rbc == 0);

    return CONTINUE;
}


// A4
int plSeparateNuclei(const cv::Mat& img, const cv::Mat& seg_open, cv::Mat& seg_nonoverlap, 
    int minSizePl, int watershedConnectivity) {
    
    // bwareaopen is done as a area threshold.
    int compcount2;
    cv::Mat seg_big_t = serial::bwareaopen2(seg_open, false, true, minSizePl, 
        std::numeric_limits<int>::max(), 8, compcount2);

    cv::Mat disk3 = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));

    cv::Mat seg_big = cv::Mat::zeros(seg_big_t.size(), seg_big_t.type());
    dilate(seg_big_t, seg_big, disk3);

    // distance transform:  matlab code is doing this:
    // invert the image so nuclei candidates are holes
    // compute the distance (distance of nuclei pixels to background)
    // negate the distance.  so now background is still 0, but nuclei pixels have negative distances
    // set background to -inf

    // really just want the distance map.  CV computes distance to 0.
    // background is 0 in output.
    // then invert to create basins
    cv::Mat dist(seg_big.size(), CV_32FC1);

    // opencv: compute the distance to nearest zero
    // matlab: compute the distance to the nearest non-zero
    distanceTransform(seg_big, dist, CV_DIST_L2, CV_DIST_MASK_PRECISE);
    double mmin, mmax;
    minMaxLoc(dist, &mmin, &mmax);

    // invert and shift (make sure it's still positive)
    dist = - dist;  // appears to work better this way.

    // then set the background to -inf and do imhmin
    // appears to work better with -inf as background
    cv::Mat distance(dist.size(), dist.type(), -std::numeric_limits<float>::max());
    dist.copyTo(distance, seg_big);

    // then do imhmin. (prevents small regions inside bigger regions)

    cv::Mat distance2 = serial::imhmin<float>(distance, 1.0f, 8);

    cv::Mat nuclei = cv::Mat::zeros(img.size(), img.type());
    img.copyTo(nuclei, seg_big);

    // watershed in openCV requires labels.  input foreground > 0, 0 is background
    // critical to use just the nuclei and not the whole image - else get a ring surrounding the regions.
    cv::Mat watermask = serial::watershed2(nuclei, distance2, watershedConnectivity);

    // MASK approach
    seg_nonoverlap = cv::Mat::zeros(seg_big.size(), seg_big.type());
    seg_big.copyTo(seg_nonoverlap, (watermask >= 0));

    return CONTINUE;
}

int segmentNuclei(const cv::Mat& img, cv::Mat& output,
    int &compcount, int *&bbox, unsigned char blue, unsigned char green, 
    unsigned char red, double T1, double T2, unsigned char G1, int minSize, 
    int maxSize, unsigned char G2, int minSizePl, int minSizeSeg, int maxSizeSeg, 
    int fillHolesConnectivity, int reconConnectivity, int watershedConnectivity) {
    
    // image in BGR format
    if (!img.data) {
        std::cout << "bad BGR" << std::endl;
        return -1;
    }

    cv::Mat seg_norbc;
    int findCandidateResult = plFindNucleusCandidates(img, seg_norbc, blue, 
        green, red, T1, T2, G1, minSize, maxSize, G2, fillHolesConnectivity, 
        reconConnectivity);

    if (findCandidateResult != CONTINUE) {
        std::cout << "bad candidates" << std::endl;
        return findCandidateResult;
    }


    cv::Mat seg_nohole = serial::imfillHoles<unsigned char>(seg_norbc, true, 4);

    cv::Mat disk3 = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));
    cv::Mat seg_open = serial::morphOpen<unsigned char>(seg_nohole, disk3);

    cv::Mat seg_nonoverlap;
    int sepResult = plSeparateNuclei(img, seg_open, seg_nonoverlap, minSizePl, 
        watershedConnectivity);

    if (sepResult != CONTINUE) {
        std::cout << "bad sepResult" << std::endl;
        return sepResult;
    }

    int compcount2;
    // MASK approach
    cv::Mat seg = serial::bwareaopen2(seg_nonoverlap, false, true, minSizeSeg, 
        maxSizeSeg, 4, compcount2);
    if (compcount2 == 0) {
        std::cout << "no candidates" << std::endl;
        return NO_CANDIDATES_LEFT;
    }
    
    // don't worry about bwlabel.
    // MASK approach
    cv::Mat final = serial::imfillHoles<unsigned char>(seg, true, fillHolesConnectivity);

    // MASK approach
    output = serial::bwlabel2(final, 8, true);
    final.release();

    ConnComponents cc;
    bbox = cc.boundingBox(output.cols, output.rows, (int *)output.data, 0, compcount);

    return SUCCESS;
}

int segmentNuclei(const std::string& in, const std::string& out,
    int &compcount, int *&bbox, unsigned char blue, unsigned char green, 
    unsigned char red, double T1, double T2, unsigned char G1, int minSize, 
    int maxSize, unsigned char G2, int minSizePl, int minSizeSeg, int maxSizeSeg, 
    int fillHolesConnectivity, int reconConnectivity, int watershedConnectivity) {

    cv::Mat input = cv::imread(in);
    if (!input.data) {
        std::cout << "bad input img" << std::endl;
        return INVALID_IMAGE;
    }

    cv::Mat output;

    int status = segmentNuclei(input, output, compcount, bbox, blue, green, 
        red, T1, T2, G1, minSize, maxSize, G2, minSizePl, minSizeSeg, maxSizeSeg, 
        fillHolesConnectivity, reconConnectivity, watershedConnectivity);
    input.release();

    if (status == SUCCESS)
        cv::imwrite(out, output);
    output.release();

    return status;
}
