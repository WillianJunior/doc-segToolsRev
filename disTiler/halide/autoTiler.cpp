#include "autoTiler.h"

// break input into bg and dense:
//  get hotspots through threshold
//      * can use user-defined functions
//      dilate
//  crop dense into rectangles
//  make bg rectangles
//      * merge bg's if merged is rectangular (this is done to 
//      reduce borders count)
// break dense list to reach 2*np dense tiles:
//  each dense is an initial 1 node kd-tree
//  make priority queue on the leaf nodes of all kd-trees ordered by cost
//      * use smart queue structures (fibonacci tree?)
//  until number of leaf nodes < 2*np:
//      split leaf node with highest cost
//      add new leaf nodes
//      re-sort queue

// std::string type2str(int type) {
//   std::string r;

//   uchar depth = type & CV_MAT_DEPTH_MASK;
//   uchar chans = 1 + (type >>CV_CN_SHIFT);

//   switch ( depth ) {
//     case CV_8U:  r = "8U"; break;
//     case CV_8S:  r = "8S"; break;
//     case CV_16U: r = "16U"; break;
//     case CV_16S: r = "16S"; break;
//     case CV_32S: r = "32S"; break;
//     case CV_32F: r = "32F"; break;
//     case CV_64F: r = "64F"; break;
//     default:     r = "User"; break;
//   }

//   r += "C";
//   r += (chans+'0');

//   return r;
// }

// overloaded < operator for sorting rect_t by x coordinate
bool operator<(const rect_t& a, const rect_t& b) {return a.xi < b.xi;}
// bool operator==(const rect_t& a, const rect_t& b) {
//     return a.xi == b.xi && a.xo == b.xo && a.yi == b.yi && a.yo == b.yo;
// }

// yi = oldY
// yo = min(r,cur)
// xo = width
void newBlocks(std::list<rect_t>& out, std::multiset<rect_t> cur, 
    int yi, int yo, int xo) {

    int xi = 0;
    for (rect_t r : cur) {
        rect_t newR = {xi, yi, r.xi, yo};
        out.push_back(newR);
        xi = r.xo;
    }

    // add last
    rect_t newR = {xi, yi, xo, yo};
    out.push_back(newR);
}

std::list<rect_t> autoTiler(cv::Mat& input, int border, 
    int bgThreshold, int erosionSize) {

    std::cout << "starting tiling" << std::endl;

    std::list<rect_t> output;
    
    // create a binary threshold mask of the input and dilate it
    cv::Mat mask;
    cv::threshold(input, mask, bgThreshold, CV_MAX_PIX_VAL, CV_THR_BIN_INV); 
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
        cv::Size(2*erosionSize + 1, 2*erosionSize+1));
    cv::dilate(mask, mask, element);

    // cv::imwrite("./mask2.png", mask); // OK

    // extract rectangular regions from mask
    cv::connectedComponents(mask, mask);
    cv::Mat cm = mask.clone();
    cm.convertTo(cm, CV_8U);
    cv::applyColorMap(cm, cm, cv::COLORMAP_JET);
    cv::imwrite("./mask3.png", cm);

    double maxLabel;
    cv::minMaxLoc(mask, NULL, &maxLabel);
    std::cout << "img size: " << mask.cols << "x" << mask.rows << std::endl;
    std::cout << "labels: " << maxLabel << std::endl;
    
    rect_t denseArr[(int)maxLabel]; // i.e. a thread safe list
    #pragma omp parallel for // toooo slow to have it not in parallel
    for (int l=1; l<(int)maxLabel; l++) { // l starts at 1 since 0 is background
        cv::Mat subMask = cv::Mat::zeros(mask.size(), mask.type());
        cv::Mat subMaskTmp = cv::Mat::zeros(mask.size(), mask.type());

        std::cout << "finding submask points of label " << l << std::endl;
        // subMaskTmp = mask - l
        cv::subtract(mask, cv::Scalar(l), subMaskTmp);
        // subMask(I) = 255 if subMaskTmp(I) != 0
        subMaskTmp = cv::abs(subMaskTmp);
        subMaskTmp.convertTo(subMaskTmp, CV_8U);
        cv::add(subMask, cv::Scalar(CV_MAX_PIX_VAL), subMask, subMaskTmp);
        // subMask = 255 - subMask
        cv::subtract(cv::Scalar(CV_MAX_PIX_VAL), subMask, subMask);
        
        // create a minimum rectangle area of all pixels of the submask
        // std::cout << "getting max rectangle" << std::endl;
        subMask.convertTo(subMask, CV_8U);
        cv::Rect r = cv::boundingRect(subMask);
        rect_t rr = {.xi=r.x, .yi=r.y, .xo=r.x+r.width, .yo=r.y+r.height};
        denseArr[l] = rr;
    }

    // generate the lists from the dense array
    std::list<rect_t> dense;
    dense = std::list<rect_t>(denseArr, 
        denseArr + sizeof(denseArr)/sizeof(rect_t));
    output = std::list<rect_t>(denseArr, 
        denseArr + sizeof(denseArr)/sizeof(rect_t));

    // sort the list of dense regions by its y coordinate (using lambda)
    dense.sort([](const rect_t& a, const rect_t& b) { return a.yi < b.yi;});

    // generate the background regions
    std::cout << "generating bg regions" << std::endl;
    int oldY = 0;
    std::multiset<rect_t> cur;
    while (!cur.empty() || !dense.empty()) {
        std::cout << "|<dense,cur>| = " << dense.size() << ", " 
            << cur.size() << std::endl;
        
        // get the heads of both lists
        rect_t r;
        if (!dense.empty())
            r = *(dense.begin());

        // check if the current y is from the beginning of the end of a rect
        // two comparisons are necessary since there may be a beginning with
        // an end on the same coordinate
        if (cur.empty() || (!dense.empty() && r.yi < cur.begin()->yo)) {
            // make the new rect blocks
            // newBlocks(output, cur, oldY, r.yi, input.cols);
            cur.insert(r);
            dense.erase(dense.begin());
        }
        if (dense.empty() || (!cur.empty() && r.yi > cur.begin()->yo)) {
            // make the new rect blocks
            // newBlocks(output, cur, oldY, cur.begin()->yo, input.cols);
            cur.erase(cur.begin());
        }
    }

    // add a border to all rect regions
    std::cout << output.size() << " regions to process" << std::endl;
    cv::Mat final = input.clone();
    for (std::list<rect_t>::iterator r=output.begin(); r!=output.end(); r++) {
        // r->xi = std::min(r->xi-border, 0);
        // r->xo = std::max(r->xo+border, input.cols);
        // r->yi = std::min(r->yi-border, 0);
        // r->yo = std::max(r->yo+border, input.rows);

        // std:: cout << "(" << r->xi << "," << r->yi << "), (" 
        //     << r->xo << "," << r->yo << ")" << std::endl;

        // draw areas for verification
        cv::rectangle(final, cv::Point(r->xi,r->yi), 
            cv::Point(r->xo,r->yo),(0,0,0),10);
    }
    cv::imwrite("./maskf.png", final);

    return output;
}
