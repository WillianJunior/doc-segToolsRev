#include <iostream>
#include <ctime>

#include "Halide.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "pixelsOps.hpp"

#define CONTINUE 0
#define BACKGROUND 1
#define BACKGROUND_LIKELY 2
#define NO_CANDIDATES_LEFT 3
#define SUCCESS 4
#define INVALID_IMAGE 5

void display(cv::Mat img, std::string txt) {
    namedWindow(txt, cv::WINDOW_AUTOSIZE);
    imshow(txt, img);
}

void waitKey() {
    int a;
    std::cout << "waiting key" << std::endl;
    std::cin >> a;
}

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

    // cv::imwrite("background.tiff", background);

    // waitKey();

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
    
    std::clock_t t0 = std::clock();

    // bwareaopen is done as a area threshold.
    int compcount2;
    cv::Mat seg_big_t = serial::bwareaopen2(seg_open, false, true, minSizePl, 
        std::numeric_limits<int>::max(), 8, compcount2);
    std::clock_t t1 = std::clock();
    double elapsed_secs = double(t1-t0)/CLOCKS_PER_SEC;
    std::cout << "bwareaopen2: " << elapsed_secs << std::endl;

    cv::Mat disk3 = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));
    std::clock_t t2 = std::clock();
    elapsed_secs = double(t2-t1)/CLOCKS_PER_SEC;
    std::cout << "getStructuringElement: " << elapsed_secs << std::endl;

    cv::Mat seg_big = cv::Mat::zeros(seg_big_t.size(), seg_big_t.type());
    dilate(seg_big_t, seg_big, disk3);
    std::clock_t t3 = std::clock();
    elapsed_secs = double(t3-t2)/CLOCKS_PER_SEC;
    std::cout << "dilate: " << elapsed_secs << std::endl;

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

    std::clock_t t4 = std::clock();
    elapsed_secs = double(t4-t3)/CLOCKS_PER_SEC;
    std::cout << "preWs: " << elapsed_secs << std::endl;

    // then do imhmin. (prevents small regions inside bigger regions)

    cv::Mat distance2 = serial::imhmin<float>(distance, 1.0f, 8);
    std::clock_t t5 = std::clock();
    elapsed_secs = double(t5-t4)/CLOCKS_PER_SEC;
    std::cout << "imhmin: " << elapsed_secs << std::endl;

    cv::Mat nuclei = cv::Mat::zeros(img.size(), img.type());
    img.copyTo(nuclei, seg_big);

    std::clock_t t6 = std::clock();
    elapsed_secs = double(t6-t5)/CLOCKS_PER_SEC;
    std::cout << "copyTo: " << elapsed_secs << std::endl;

    // watershed in openCV requires labels.  input foreground > 0, 0 is background
    // critical to use just the nuclei and not the whole image - else get a ring surrounding the regions.
    cv::Mat watermask = serial::watershed2(nuclei, distance2, watershedConnectivity);
    std::clock_t t7 = std::clock();
    elapsed_secs = double(t7-t6)/CLOCKS_PER_SEC;
    std::cout << "watershed2: " << elapsed_secs << std::endl;

    // MASK approach
    seg_nonoverlap = cv::Mat::zeros(seg_big.size(), seg_big.type());
    seg_big.copyTo(seg_nonoverlap, (watermask >= 0));
    std::clock_t t8 = std::clock();
    elapsed_secs = double(t8-t7)/CLOCKS_PER_SEC;
    std::cout << "copyTo: " << elapsed_secs << std::endl;

    return CONTINUE;
}

int segmentNuclei(const cv::Mat& img, cv::Mat& output,
    int &compcount, int *&bbox, unsigned char blue, unsigned char green, 
    unsigned char red, double T1, double T2, unsigned char G1, int minSize, 
    int maxSize, unsigned char G2, int minSizePl, int minSizeSeg, int maxSizeSeg, 
    int fillHolesConnectivity, int reconConnectivity, int watershedConnectivity) {

    double elapsed_secs;
    std::clock_t t0 = std::clock();
    
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

    std::clock_t t1 = std::clock();
    elapsed_secs = double(t1-t0)/CLOCKS_PER_SEC;
    std::cout << "plFindNucleusCandidates: " << elapsed_secs << std::endl;

    cv::Mat seg_nohole = serial::imfillHoles<unsigned char>(seg_norbc, true, 4);
    std::clock_t t2 = std::clock();
    elapsed_secs = double(t2-t1)/CLOCKS_PER_SEC;
    std::cout << "imfillHoles: " << elapsed_secs << std::endl;

    cv::Mat disk3 = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));
    std::clock_t t3 = std::clock();
    elapsed_secs = double(t3-t2)/CLOCKS_PER_SEC;
    std::cout << "getStructuringElement: " << elapsed_secs << std::endl;

    cv::Mat seg_open = serial::morphOpen<unsigned char>(seg_nohole, disk3);
    std::clock_t t4 = std::clock();
    elapsed_secs = double(t4-t3)/CLOCKS_PER_SEC;
    std::cout << "morphOpen: " << elapsed_secs << std::endl;

    cv::Mat seg_nonoverlap;
    int sepResult = plSeparateNuclei(img, seg_open, seg_nonoverlap, minSizePl, 
        watershedConnectivity);
    std::clock_t t5 = std::clock();
    elapsed_secs = double(t5-t4)/CLOCKS_PER_SEC;
    std::cout << "plSeparateNuclei: " << elapsed_secs << std::endl;

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
    std::clock_t t6 = std::clock();
    elapsed_secs = double(t6-t5)/CLOCKS_PER_SEC;
    std::cout << "bwareaopen2: " << elapsed_secs << std::endl;
    
    // don't worry about bwlabel.
    // MASK approach
    cv::Mat final = serial::imfillHoles<unsigned char>(seg, true, fillHolesConnectivity);
    std::clock_t t7 = std::clock();
    elapsed_secs = double(t7-t6)/CLOCKS_PER_SEC;
    std::cout << "imfillHoles: " << elapsed_secs << std::endl;

    // MASK approach
    output = serial::bwlabel2(final, 8, true);
    final.release();
    std::clock_t t8 = std::clock();
    elapsed_secs = double(t8-t7)/CLOCKS_PER_SEC;
    std::cout << "bwlabel2: " << elapsed_secs << std::endl;

    ConnComponents cc;
    bbox = cc.boundingBox(output.cols, output.rows, (int *)output.data, 0, compcount);
    std::clock_t t9 = std::clock();
    elapsed_secs = double(t9-t8)/CLOCKS_PER_SEC;
    std::cout << "boundingBox: " << elapsed_secs << std::endl;

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