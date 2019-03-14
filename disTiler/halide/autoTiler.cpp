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
// bool operator<(const rect_t& a, const rect_t& b) {return a.xi < b.xi;}
bool operator==(const rect_t& a, const rect_t& b) {
    return a.xi == b.xi && a.xo == b.xo && a.yi == b.yi && a.yo == b.yo;
}

struct rect_tCompX{
    bool operator()(const rect_t& a, const rect_t& b) {
        return a.xi < b.xi || (a.xi == b.xi && 
            (a.xo != b.xo || a.yi != b.yi || a.yo != b.yo));
    }
};

struct rect_tCompY{
    bool operator()(const rect_t& a, const rect_t& b) {
        return a.yo < b.yo || (a.yo == b.yo && 
            (a.xo != b.xo || a.yi != b.yi || a.xi != b.xi));
    }
};

void printMS(std::multiset<rect_t,rect_tCompX> l) {
    for (rect_t r : l)
        std::cout << r.xi << "," << r.yi << "," 
            << r.xo << "," << r.yo << std::endl;
}

void printMS(std::multiset<rect_t,rect_tCompY> l) {
    for (rect_t r : l)
        std::cout << r.xi << "," << r.yi << "," 
            << r.xo << "," << r.yo << std::endl;
}

void printMS(std::multiset<rect_t,rect_tCompX> lx,
    std::multiset<rect_t,rect_tCompY> ly) {

    std::multiset<rect_t,rect_tCompX>::iterator ix = lx.begin();
    std::multiset<rect_t,rect_tCompY>::iterator iy = ly.begin();

    std::cout << "X \t\t|\t\t Y" << std::endl;
    while (ix != lx.end() || iy != ly.end()) {
        if (ix == lx.end()) {
            std::cout << "---\t\t\t";
        } else {
            std::cout << ix->xi << "," << ix->yi << "|" 
                << ix->xo << "," << ix->yo << "\t\t";
            ix++;
        }
        if (iy == ly.end()) {
            std::cout << "---\t\t\t";
        } else {
            std::cout << iy->yi << "," << iy->yi << "|" 
                << iy->yo << "," << iy->yo << std::endl;
            iy++;
        }
    }
}

inline bool isInside(int x, int y, rect_t r2) {
    return r2.xi <= x && r2.yi <= y && r2.xo >= x && r2.yo >= y;
}

inline bool isInside(rect_t r1, rect_t r2) {
    return isInside(r1.xi, r1.yi, r2) && isInside(r1.xo, r1.yo, r2);
}

void removeInsideOvlp(std::list<rect_t>& output) {
    // first create an array for parallelization
    int outS = output.size();
    rect_t outArray[outS];
    std::copy(output.begin(), output.end(), outArray);

    // create an array of tags for non repeated regions
    bool outArrayNR[outS];
    for (int i=0; i<outS; i++) {outArrayNR[i] = false;}

    // compare each element with each other, checking if it's inside any
    #pragma omp parallel for
    for (int i=0; i<outS; i++) {
        int j;
        for (j=0; j<outS; j++) {
            if (isInside(outArray[i], outArray[j]) && i!=j)
                break;
        }
        // check if no other bigger region was found
        if (j == outS) {
            outArrayNR[i] = true;
        }
    }

    // clear the old elements and add only the unique regions
    output.clear();
    for (int i=0; i<outS; i++) {
        if (outArrayNR[i])
            output.push_back(outArray[i]);
    }
}

inline bool hOvlp(rect_t big, rect_t small) {
    return big.yi <= small.yi && big.yo >= small.yo 
        && ((big.xi < small.xi && big.xo > small.xi)
        || (big.xi < small.xo && big.xo > small.xo));
}

inline bool vOvlp(rect_t big, rect_t small) {
    return big.xi <= small.xi && big.xo >= small.xo 
        && ((big.yi < small.yi && big.yo > small.yi)
        || (big.yi < small.yo && big.yo > small.yo));
}

void removeSideOvlp(std::list<rect_t>& output) {
    // first create an array for parallelization
    int outS = output.size();
    rect_t outArray[outS];
    std::copy(output.begin(), output.end(), outArray);

    // compare each element with each other, checking if it's inside any
    #pragma omp parallel for
    for (int i=0; i<outS; i++) {
        int big, small;
        for (int j=0; j<outS; j++) {
            // check if there is a horizontal overlapping
            if (hOvlp(outArray[i], outArray[j]) 
                || hOvlp(outArray[i], outArray[j])) {

                // find which is the big one
                big = i; small = j;
                if (outArray[i].yi > outArray[j].yi) {
                    big = j; small = i;
                }

                // remove the overlapping of small with big
                if (outArray[small].xi > outArray[big].xi) // big left
                    outArray[small].xi = outArray[big].xo;
                else // big on the right
                    outArray[small].xo = outArray[big].xi;
            }
            // check if there is a vertical overlapping
            if (vOvlp(outArray[i], outArray[j]) 
                || vOvlp(outArray[i], outArray[j])) {

                // find which is the big one
                big = i; small = j;
                if (outArray[i].xi > outArray[j].xi) {
                    big = j; small = i;
                }

                // remove the overlapping of small with big
                if (outArray[small].yi > outArray[big].yi) // big up
                    outArray[small].yi = outArray[big].yo;
                else // big is the down region
                    outArray[small].yo = outArray[big].yi;
            }
        }
    }

    // clear the old elements and add only the unique regions
    output.clear();
    for (int i=0; i<outS; i++) {
        output.push_back(outArray[i]);
    }
}

inline int area (rect_t r) {
    return (r.xo-r.xi) * (r.yo-r.yi);
}

void removeDiagOvlp(std::list<rect_t>& ovlpCand, std::list<rect_t>& nonMod) {
    // first create an array for parallelization
    int outS = ovlpCand.size();
    rect_t outArray[outS];
    std::copy(ovlpCand.begin(), ovlpCand.end(), outArray);

    // array for the replaced 3 new blocks for every initial block
    rect_t outRepArray[outS][3];

    // marker array for showing which blocks were to be replaced
    bool outRepArrayR[outS];
    bool outRepArrayS[outS];
    for (int i=0; i<outS; i++) {
        outRepArrayR[i] = false;
        outRepArrayS[i] = false;
    }

    // compare each element with each other, checking if it's inside any
    #pragma omp parallel for
    for (int i=0; i<outS; i++) {
        for (int j=0; j<outS; j++) {
            rect_t b = outArray[i];
            rect_t s = outArray[j];

            // break the big block if there is a diagonal overlap
            if (b.xi > s.xi && b.xi < s.xo && b.yo > s.yi && b.yo < s.yo) {
                // big on upper right diagonal
                // only break down if <i> is the big block of a diagonal overlap
                if (area(outArray[i]) < area(outArray[j])) {
                    outRepArrayS[i] = true;
                    continue;
                }
                outRepArrayR[i] = true;
                outRepArray[i][0] = {b.xi, b.yi, s.xo, s.yi};
                outRepArray[i][1] = {s.xo, b.yi, b.xo, s.yi};
                outRepArray[i][2] = {s.xo, s.yi, b.xo, b.yo};
            } else if (b.xo > s.xi && b.xo < s.xo && b.yo > s.yi && b.yo < s.yo) {
                // big on upper left
                // only break down if <i> is the big block of a diagonal overlap
                if (area(outArray[i]) < area(outArray[j])) {
                    outRepArrayS[i] = true;
                    continue;
                }
                outRepArrayR[i] = true;
                outRepArray[i][0] = {b.xi, b.yi, s.xi, s.yi};
                outRepArray[i][1] = {s.xi, b.yi, b.xo, s.yi};
                outRepArray[i][2] = {b.xi, s.yi, s.xi, b.yo};
            } else if (b.xi > s.xi && b.xi < s.xo && b.yi > s.yi && b.yi < s.yo) {
                // big on bottom right
                // only break down if <i> is the big block of a diagonal overlap
                if (area(outArray[i]) < area(outArray[j])) {
                    outRepArrayS[i] = true;
                    continue;
                }
                outRepArrayR[i] = true;
                outRepArray[i][0] = {s.xo, b.yi, b.xo, s.yo};
                outRepArray[i][1] = {b.xi, s.yo, s.xo, b.yo};
                outRepArray[i][2] = {s.xo, s.yo, b.xo, b.yo};
            } else if (b.xo > s.xi && b.xo < s.xo && b.yi > s.yi && b.yi < s.yo) {
                // big on bottom left
                // only break down if <i> is the big block of a diagonal overlap
                if (area(outArray[i]) < area(outArray[j])) {
                    outRepArrayS[i] = true;
                    continue;
                }
                outRepArrayR[i] = true;
                outRepArray[i][0] = {b.xi, b.yi, s.xi, s.yo};
                outRepArray[i][1] = {s.xi, s.yo, b.xo, b.yo};
                outRepArray[i][2] = {b.xi, s.yo, s.xi, b.yo};
            }
        }
    }

    // clear the old elements and add only the unique regions
    ovlpCand.clear();
    for (int i=0; i<outS; i++) {
        if (outRepArrayR[i]) {
            ovlpCand.push_back(outRepArray[i][0]);
            ovlpCand.push_back(outRepArray[i][1]);
            ovlpCand.push_back(outRepArray[i][2]);
        } else if (outRepArrayS[i])
            ovlpCand.push_back(outArray[i]);
        else
            nonMod.push_back(outArray[i]);

    }
}

// yi = oldY
// yo = min(r,cur)
// xo = width
void newBlocks(std::list<rect_t>& out, std::multiset<rect_t,rect_tCompX> cur, 
    int yi, int yo, int xo) {

    int xi = 0;
    for (rect_t r : cur) {
        rect_t newR = {xi, yi, r.xi, yo};
        // don't add lines i.e. zero area rect
        if (newR.xi != newR.xo && newR.yi != newR.yo)
            out.push_back(newR);
        xi = r.xo;
    }

    // add last
    rect_t newR = {xi, yi, xo, yo};
    if (newR.xi != newR.xo && newR.yi != newR.yo)
        out.push_back(newR);
}

void generateBackground(std::list<rect_t>& dense, 
    std::list<rect_t>& output, int maxCols, cv::Mat& input) {

    std::cout << "generating bg regions" << std::endl;
    int oldY = 0;
    std::multiset<rect_t,rect_tCompX> curX;
    std::multiset<rect_t,rect_tCompY> curY;

    // sort dense by the top of each area
    dense.sort([](const rect_t& a, const rect_t& b) { return a.yi < b.yi;});

    while (!curY.empty() || !dense.empty()) {
        // std::cout << "|<dense,cur>| = " << dense.size() << ", " 
        //     << cur.size() << std::endl;
        
        // get the heads of both lists
        rect_t r;
        if (!dense.empty())
            r = *(dense.begin());

        // std::cout << "dense|cur: (<" << r.xi << "," << r.yi << ">,<" 
        //         << r.xo << "," << r.yo << ">)" << std::endl;
        // if (!cur.empty() && !dense.empty()) {
        //     rect_t c = *(cur.begin());
        //     std::cout << "dense|cur: (<" << r.xi << "," << r.yi << ">,<" 
        //         << r.xo << "," << r.yo << ">) | (<" << c.xi << "," << c.yi 
        //         << ">,<" << c.xo << "," << c.yo << ">)" << std::endl;
        //     if (r.xi == 3337 && r.yi == 192) {
        //         int cc;
        //         std::cin >> cc;
        //     }
        // }

        // check if the current y is from the beginning of the end of a rect
        // two comparisons are necessary since there may be a beginning with
        // an end on the same coordinate
        // std::cout << "if1: " << r.yi << " <= "
        //     << (curY.size()>0 ? std::to_string(curY.begin()->yo) : " empty") 
        //     << std::endl;
        if (curY.empty() || (!dense.empty() && r.yi <= curY.begin()->yo)) {
            // make the new rect blocks
            newBlocks(output, curX, oldY, r.yi, maxCols);

            // update cur structure
            oldY = r.yi;
            curX.emplace(r);
            curY.emplace(r);

            // remove the new node from dense
            dense.erase(dense.begin());
            dense.sort([](const rect_t& a, const rect_t& b) 
                {return a.yi < b.yi;});

            // std::cout << r.xi << "," << r.yi << "," << r.xo 
            //     << "," << r.yo << " was new" << std::endl;
        } 

        // std::cout << "if2: " << r.yi << " >= "
        //     << (curY.size()>0 ? std::to_string(curY.begin()->yo) : " empty") 
        //     << std::endl;
        else if (dense.empty() || (!dense.empty() && r.yi >= curY.begin()->yo)) {
            // make the new rect blocks
            newBlocks(output, curX, oldY, curY.begin()->yo, maxCols);

            // update cur structure and remove top cur
            oldY = curY.begin()->yo;
            // std::cout << curY.begin()->xi << "," << curY.begin()->yi 
            //     << "," << curY.begin()->xo << "," << curY.begin()->yo 
            //     << " was old" << std::endl;
            curX.erase(*curY.begin());
            curY.erase(curY.begin());
        }

        // printMS(curX, curY);
        // printMS(curX);

        // std::cout << "dense|cur: " << dense.size() << "|" 
        //     << curY.size() << "|" << curX.size() 
        //     << "-------------------------------------------" 
        //     << std::endl;

        // // add a border to all rect regions
        // cv::Mat final = input.clone();
        // int ii=0;
        // for (std::list<rect_t>::iterator r=output.begin(); 
        //     r!=output.end(); r++) {

        //     // draw areas for verification
        //     cv::rectangle(final, cv::Point(r->xi,r->yi), 
        //         cv::Point(r->xo,r->yo),(0,0,0),3);
        //     cv::Point center(r->xi + (r->xo-r->xi)/2, r->yi + (r->yo-r->yi)/2);
        //     putText(final, std::to_string(ii++), center, 1, 5, 
        //         cv::Scalar(0), 3, 0);
        // }
        // cv::imwrite("./maskf.png", final);
        // int i;
        // std::cin >> i;
    }
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

    // merge regions of interest together and mark them separately
    cv::Mat stats, centroids;
    cv::connectedComponentsWithStats(mask, mask, stats, centroids);

    // std::cout << "STATS:" << std::endl;
    // std::cout << "CC_STAT_LEFT: "   << stats.at<int>(10, cv::CC_STAT_LEFT)
    //         << "\nCC_STAT_TOP: "    << stats.at<int>(10, cv::CC_STAT_TOP)
    //         << "\nCC_STAT_WIDTH: "  << stats.at<int>(10, cv::CC_STAT_WIDTH)
    //         << "\nCC_STAT_HEIGHT: " << stats.at<int>(10, cv::CC_STAT_HEIGHT)
    //         << "\nCC_STAT_AREA: "   << stats.at<int>(10, cv::CC_STAT_AREA)
    //         << std::endl;

    // cv::Mat cm = mask.clone();
    // cm.convertTo(cm, CV_8U);
    // cv::applyColorMap(cm, cm, cl::COLORMAP_JET);
    // cv::imwrite("./mask3.png", cm);

    double maxLabel;
    cv::minMaxLoc(mask, NULL, &maxLabel);
    std::cout << "img size: " << mask.cols << "x" << mask.rows << std::endl;
    std::cout << "labels: " << maxLabel << std::endl;
    
    // generate the list of dense areas
    std::list<rect_t> ovlpCand;
    for (int i=1; i<=maxLabel; i++) { // i=1 ignores background
        int xi = stats.at<int>(i, cv::CC_STAT_LEFT);
        int yi = stats.at<int>(i, cv::CC_STAT_TOP);
        int xw = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int yh = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        rect_t rr = {.xi=xi, .yi=yi, .xo=xi+xw, .yo=yi+yh};
        ovlpCand.push_back(rr);
    }

    // keep trying to remove overlapping regions until there is none
    while (!ovlpCand.empty()) {
        // remove regions that are overlapping within another bigger region
        removeInsideOvlp(ovlpCand);

        // remove the overlap of two regions, side by side (vert and horz)
        removeSideOvlp(ovlpCand);

        // remove the diagonal overlaps
        removeDiagOvlp(ovlpCand, output);
    }
    
    // sort the list of dense regions by its y coordinate (using lambda)
    std::list<rect_t> dense(output);
    
    // generate the background regions
    generateBackground(dense, output, input.cols, input);

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
            cv::Point(r->xo,r->yo),(0,0,0),3);
    }
    cv::imwrite("./maskf.png", final);

    return output;
}
