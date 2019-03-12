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

    std::list<rect_t> output;
    
    // create a binary threshold mask of the input and dilate it
    cv::Mat mask;
    cv::threshold(input, mask, bgThreshold, CV_MAX_PIX_VAL, CV_THR_BIN); 
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
        cv::Size(2*erosionSize + 1, 2*erosionSize+1));
    cv::dilate(mask, mask, element);

    // extract rectangular regions from mask
    cv::connectedComponents(mask, mask);
    double maxLabel;
    cv::minMaxLoc(mask, NULL, &maxLabel);
    std::list<rect_t> dense;
    for (int l=1; l<maxLabel; l++) { // l starts at 1 since 0 is background
        // naive way to extract a single labeled submask
        cv::Mat subMask = cv::Mat::zeros(mask.size(), mask.type());
        for (int i = 0; i < mask.cols; i++) {
            for (int j = 0; j < mask.rows; j++) {
                if (mask.at<int>(j, i) == l) {  
                    subMask.at<int>(j, i) = CV_MAX_PIX_VAL;
                }
            }
        }
        
        // create a minimum rectangle area of all pixels of the submask
        cv::Rect r = cv::boundingRect(subMask);
        rect_t rr = {.xi=r.x, .yi=r.y, .xo=r.width-r.x, .yo=r.height-r.y};
        dense.push_back(rr);
        output.push_back(rr);
    }

    // sort the list of dense regions by its y coordinate (using lambda)
    dense.sort([](const rect_t& a, const rect_t& b) { return a.yi < b.yi;});

    // generate the background regions
    int oldY = 0;
    std::multiset<rect_t> cur;
    while (!cur.empty() || !dense.empty()) {
        
        // get the heads of both lists
        rect_t r;
        if (!dense.empty())
            r = *(dense.begin());
        std::multiset<rect_t>::iterator curFirst = cur.begin(); // head(cur)
        int curFirstYo = cur.empty() ? 0 : cur.begin()->yo; // head(cur)

        // check if the current y is from the beginning of the end of a rect
        // two comparisons are necessary since there may be a beginning with
        // an end on the same coordinate
        if (!dense.empty() && r.yi < curFirstYo) {
            // make the new rect blocks
            newBlocks(output, cur, oldY, r.yi, input.cols);
            cur.insert(r);
        }
        if (dense.empty() || r.yi > curFirstYo) {
            // make the new rect blocks
            newBlocks(output, cur, oldY, curFirst->yo, input.cols);
            cur.erase(cur.begin());
        }
    }

    // empty out the remaining of the cur list

    // add a border to all rect regions
    for (std::list<rect_t>::iterator r=output.begin(); r!=output.end(); r++) {
        r->xi = std::min(r->xi-border, 0);
        r->xo = std::max(r->xo+border, input.cols);
        r->yi = std::min(r->yi-border, 0);
        r->yo = std::max(r->yo+border, input.rows);
    }

    return output;
}
