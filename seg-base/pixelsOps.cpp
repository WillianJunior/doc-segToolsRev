#include "pixelsOps.hpp"

namespace serial {

template <typename T>
cv::Mat invert(const cv::Mat &img) {
    // write the raw image
    CV_Assert(img.channels() == 1);

    if (std::numeric_limits<T>::is_integer) {

        if (std::numeric_limits<T>::is_signed) {
            cv::Mat output;
            bitwise_not(img, output);
            return output + 1;
        } else {
            // unsigned int
            return std::numeric_limits<T>::max() - img;
        }

    } else {
        // floating point type
        return -img;
    }
}


template <typename T>
inline void propagate(const cv::Mat& image, cv::Mat& output, std::queue<int>& xQ, 
    std::queue<int>& yQ, int x, int y, T* iPtr, T* oPtr, const T& pval) {

    T qval = oPtr[x];
    T ival = iPtr[x];
    if ((qval < pval) && (ival != qval)) {
        oPtr[x] = cv::min(pval, ival);
        xQ.push(x);
        yQ.push(y);
    }
}

template <typename T>
cv::Mat imreconstruct(const cv::Mat& seeds, const cv::Mat& image, int connectivity) {
    CV_Assert(image.channels() == 1);
    CV_Assert(seeds.channels() == 1);


    cv::Mat output(seeds.size() + cv::Size(2,2), seeds.type());
    copyMakeBorder(seeds, output, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::Mat input(image.size() + cv::Size(2,2), image.type());
    copyMakeBorder(image, input, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);

    T pval, preval;
    int xminus, xplus, yminus, yplus;
    int maxx = output.cols - 1;
    int maxy = output.rows - 1;
    std::queue<int> xQ;
    std::queue<int> yQ;
    T* oPtr;
    T* oPtrMinus;
    T* oPtrPlus;
    T* iPtr;
    T* iPtrPlus;
    T* iPtrMinus;

    // raster scan
    for (int y = 1; y < maxy; ++y) {

        oPtr = output.ptr<T>(y);
        oPtrMinus = output.ptr<T>(y-1);
        iPtr = input.ptr<T>(y);

        preval = oPtr[0];
        for (int x = 1; x < maxx; ++x) {
            xminus = x-1;
            xplus = x+1;
            pval = oPtr[x];

            // walk through the neighbor pixels, left and up (N+(p)) only
            pval = cv::max(pval, cv::max(preval, oPtrMinus[x]));

            if (connectivity == 8) {
                pval = cv::max(pval, cv::max(oPtrMinus[xplus], oPtrMinus[xminus]));
            }
            preval = cv::min(pval, iPtr[x]);
            oPtr[x] = preval;
        }
    }

    // anti-raster scan
    int count = 0;
    for (int y = maxy-1; y > 0; --y) {
        oPtr = output.ptr<T>(y);
        oPtrPlus = output.ptr<T>(y+1);
        oPtrMinus = output.ptr<T>(y-1);
        iPtr = input.ptr<T>(y);
        iPtrPlus = input.ptr<T>(y+1);

        preval = oPtr[maxx];
        for (int x = maxx-1; x > 0; --x) {
            xminus = x-1;
            xplus = x+1;

            pval = oPtr[x];

            // walk through the neighbor pixels, right and down (N-(p)) only
            pval = cv::max(pval, cv::max(preval, oPtrPlus[x]));

            if (connectivity == 8) {
                pval = cv::max(pval, cv::max(oPtrPlus[xplus], oPtrPlus[xminus]));
            }

            preval = cv::min(pval, iPtr[x]);
            oPtr[x] = preval;

            // capture the seeds
            // walk through the neighbor pixels, right and down (N-(p)) only
            pval = oPtr[x];

            if ((oPtr[xplus] < cv::min(pval, iPtr[xplus])) ||
                    (oPtrPlus[x] < cv::min(pval, iPtrPlus[x]))) {
                xQ.push(x);
                yQ.push(y);
                ++count;
                continue;
            }

            if (connectivity == 8) {
                if ((oPtrPlus[xplus] < cv::min(pval, iPtrPlus[xplus])) ||
                        (oPtrPlus[xminus] < cv::min(pval, iPtrPlus[xminus]))) {
                    xQ.push(x);
                    yQ.push(y);
                    ++count;
                    continue;
                }
            }
        }
    }

    // now process the queue.
    int x, y;
    count = 0;
    while (!(xQ.empty())) {
        ++count;
        x = xQ.front();
        y = yQ.front();
        xQ.pop();
        yQ.pop();
        xminus = x-1;
        xplus = x+1;
        yminus = y-1;
        yplus = y+1;

        oPtr = output.ptr<T>(y);
        oPtrPlus = output.ptr<T>(yplus);
        oPtrMinus = output.ptr<T>(yminus);
        iPtr = input.ptr<T>(y);
        iPtrPlus = input.ptr<T>(yplus);
        iPtrMinus = input.ptr<T>(yminus);

        pval = oPtr[x];

        // look at the 4 connected components
        if (y > 0) {
            propagate<T>(input, output, xQ, yQ, x, yminus, 
                iPtrMinus, oPtrMinus, pval);
        }
        if (y < maxy) {
            propagate<T>(input, output, xQ, yQ, x, yplus, 
                iPtrPlus, oPtrPlus,pval);
        }
        if (x > 0) {
            propagate<T>(input, output, xQ, yQ, xminus, y, 
                iPtr, oPtr,pval);
        }
        if (x < maxx) {
            propagate<T>(input, output, xQ, yQ, xplus, y, 
                iPtr, oPtr,pval);
        }

        // now 8 connected
        if (connectivity == 8) {

            if (y > 0) {
                if (x > 0) {
                    propagate<T>(input, output, xQ, yQ, xminus, 
                        yminus, iPtrMinus, oPtrMinus, pval);
                }
                if (x < maxx) {
                    propagate<T>(input, output, xQ, yQ, xplus, 
                        yminus, iPtrMinus, oPtrMinus, pval);
                }

            }
            if (y < maxy) {
                if (x > 0) {
                    propagate<T>(input, output, xQ, yQ, xminus, 
                        yplus, iPtrPlus, oPtrPlus,pval);
                }
                if (x < maxx) {
                    propagate<T>(input, output, xQ, yQ, xplus, 
                        yplus, iPtrPlus, oPtrPlus,pval);
                }

            }
        }
    }

    return output(cv::Range(1, maxy), cv::Range(1, maxx));

}

// inclusive min, exclusive max
cv::Mat bwareaopen2(const cv::Mat& image, bool labeled, bool flatten, 
    int minSize, int maxSize, int connectivity, int& count) {

    // only works for binary images.
    CV_Assert(image.channels() == 1);
    // only works for binary images.
    if (labeled == false)
        CV_Assert(image.type() == CV_8U);
    else
        CV_Assert(image.type() == CV_32S);

    //copy, to make data continuous.
    cv::Mat input = cv::Mat::zeros(image.size(), image.type());
    image.copyTo(input);
    cv::Mat_<int> output = cv::Mat_<int>::zeros(input.size());

    ConnComponents cc;
    if (labeled == false) {
        cv::Mat_<int> temp = cv::Mat_<int>::zeros(input.size());
        cc.label((unsigned char*)input.data, input.cols, input.rows, 
            (int *)temp.data, -1, connectivity);
        count = cc.areaThresholdLabeled((int *)temp.data, temp.cols, 
            temp.rows, (int *)output.data, -1, minSize, maxSize);
        temp.release();
    } else {
        count = cc.areaThresholdLabeled((int *)input.data, input.cols, 
            input.rows, (int *)output.data, -1, minSize, maxSize);
    }

    input.release();
    if (flatten == true) {
        cv::Mat O2 = cv::Mat::zeros(output.size(), CV_8U);
        O2 = output > -1;
        output.release();
        return O2;
    } else
        return output;

}

template <typename T>
inline void propagateBinary(const cv::Mat& image, cv::Mat& output, 
    std::queue<int>& xQ, std::queue<int>& yQ, int x, int y, T* iPtr, 
    T* oPtr, const T& foreground) {

    if ((oPtr[x] == 0) && (iPtr[x] != 0)) {
        oPtr[x] = foreground;
        xQ.push(x);
        yQ.push(y);
    }
}

/** optimized serial implementation for binary,
 from Vincent paper on "Morphological Grayscale Reconstruction in Image 
 Analysis: Applicaitons and Efficient Algorithms" connectivity is either 
 4 or 8, default 4.  background is assume to be 0, foreground is assumed 
 to be NOT 0.
 */
template <typename T>
cv::Mat imreconstructBinary(const cv::Mat& seeds, 
    const cv::Mat& image, int connectivity) {

    CV_Assert(image.channels() == 1);
    CV_Assert(seeds.channels() == 1);

    cv::Mat output(seeds.size() + cv::Size(2,2), seeds.type());
    copyMakeBorder(seeds, output, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::Mat input(image.size() + cv::Size(2,2), image.type());
    copyMakeBorder(image, input, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);

    T pval, ival;
    int xminus, xplus, yminus, yplus;
    int maxx = output.cols - 1;
    int maxy = output.rows - 1;
    std::queue<int> xQ;
    std::queue<int> yQ;
    T* oPtr;
    T* oPtrPlus;
    T* oPtrMinus;
    T* iPtr;
    T* iPtrPlus;
    T* iPtrMinus;

    int count = 0;
    // contour pixel determination.  if any neighbor of a 1 pixel is 
    //   0, and the image is 1, then boundary
    for (int y = 1; y < maxy; ++y) {
        oPtr = output.ptr<T>(y);
        oPtrPlus = output.ptr<T>(y+1);
        oPtrMinus = output.ptr<T>(y-1);
        iPtr = input.ptr<T>(y);

        for (int x = 1; x < maxx; ++x) {

            pval = oPtr[x];
            ival = iPtr[x];

            if (pval != 0 && ival != 0) {
                xminus = x - 1;
                xplus = x + 1;

                // 4 connected
                if ((oPtrMinus[x] == 0) ||
                        (oPtrPlus[x] == 0) ||
                        (oPtr[xplus] == 0) ||
                        (oPtr[xminus] == 0)) {
                    xQ.push(x);
                    yQ.push(y);
                    ++count;
                    continue;
                }

                // 8 connected

                if (connectivity == 8) {
                    if ((oPtrMinus[xminus] == 0) ||
                        (oPtrMinus[xplus] == 0) ||
                        (oPtrPlus[xminus] == 0) ||
                        (oPtrPlus[xplus] == 0)) {
                                xQ.push(x);
                                yQ.push(y);
                                ++count;
                                continue;
                    }
                }
            }
        }
    }

    // now process the queue.
    T outval = std::numeric_limits<T>::max();
    int x, y;
    count = 0;
    while (!(xQ.empty())) {
        ++count;
        x = xQ.front();
        y = yQ.front();
        xQ.pop();
        yQ.pop();
        xminus = x-1;
        yminus = y-1;
        yplus = y+1;
        xplus = x+1;

        oPtr = output.ptr<T>(y);
        oPtrMinus = output.ptr<T>(y-1);
        oPtrPlus = output.ptr<T>(y+1);
        iPtr = input.ptr<T>(y);
        iPtrMinus = input.ptr<T>(y-1);
        iPtrPlus = input.ptr<T>(y+1);

        // look at the 4 connected components
        if (y > 0) {
            propagateBinary<T>(input, output, xQ, yQ, 
                x, yminus, iPtrMinus, oPtrMinus, outval);
        }
        if (y < maxy) {
            propagateBinary<T>(input, output, xQ, yQ, 
                x, yplus, iPtrPlus, oPtrPlus, outval);
        }
        if (x > 0) {
            propagateBinary<T>(input, output, xQ, yQ, 
                xminus, y, iPtr, oPtr, outval);
        }
        if (x < maxx) {
            propagateBinary<T>(input, output, xQ, yQ, 
                xplus, y, iPtr, oPtr, outval);
        }

        // now 8 connected
        if (connectivity == 8) {

            if (y > 0) {
                if (x > 0) {
                    propagateBinary<T>(input, output, xQ, yQ, 
                        xminus, yminus, iPtrMinus, oPtrMinus, outval);
                }
                if (x < maxx) {
                    propagateBinary<T>(input, output, xQ, yQ, 
                        xplus, yminus, iPtrMinus, oPtrMinus, outval);
                }

            }
            if (y < maxy) {
                if (x > 0) {
                    propagateBinary<T>(input, output, xQ, yQ, 
                        xminus, yplus, iPtrPlus, oPtrPlus,outval);
                }
                if (x < maxx) {
                    propagateBinary<T>(input, output, xQ, yQ, xplus, 
                        yplus, iPtrPlus, oPtrPlus,outval);
                }
            }
        }
    }
    return output(cv::Range(1, maxy), cv::Range(1, maxx));
}

// Operates on BINARY IMAGES ONLY
template <typename T>
cv::Mat bwselect(const cv::Mat& binaryImage, const cv::Mat& seeds, int connectivity) {
    CV_Assert(binaryImage.channels() == 1);
    CV_Assert(seeds.channels() == 1);
    
    cv::Mat marker = cv::Mat::zeros(seeds.size(), seeds.type());
    binaryImage.copyTo(marker, seeds);

    marker = imreconstructBinary<T>(marker, binaryImage, connectivity);

    return marker & binaryImage;
}

template <typename T>
cv::Mat imfillHoles(const cv::Mat& image, bool binary, int connectivity) {
    CV_Assert(image.channels() == 1);

    cv::Scalar mn;
    T mx = std::numeric_limits<T>::max();
    cv::Rect roi = cv::Rect(1, 1, image.cols, image.rows);

    // copy the input and pad with -inf.
    cv::Mat mask(image.size() + cv::Size(2,2), image.type());
    copyMakeBorder(image, mask, 1, 1, 1, 1, cv::BORDER_CONSTANT, mn);
    // create marker with inf inside and -inf at border, and take its complement
    cv::Mat marker;
    cv::Mat marker2(image.size(), image.type(), mn);
    // them make the border - OpenCV does not replicate the 
    //   values when one Mat is a region of another.
    copyMakeBorder(marker2, marker, 1, 1, 1, 1, cv::BORDER_CONSTANT, mx);

    // now do the work...
    mask = invert<T>(mask);

    cv::Mat output;
    if (binary == true) {
        output = imreconstructBinary<T>(marker, mask, connectivity);
    } else {
        output = imreconstruct<T>(marker, mask, connectivity);
    }

    output = invert<T>(output);

    return output(roi);
}

template <typename T>
cv::Mat imhmin(const cv::Mat& image, T h, int connectivity) {
    // only works for intensity images.
    CV_Assert(image.channels() == 1);

    //  IMHMIN(I,H) suppresses all minima in I whose depth is less than h
    // MatLAB implementation:
    /**
     *
        I = imcomplement(I);
        I2 = imreconstruct(imsubtract(I,h), I, conn);
        I2 = imcomplement(I2);
     *
     */
    cv::Mat mask = invert<T>(image);
    cv::Mat marker = mask - h;

    cv::Mat output = imreconstruct<T>(marker, mask, connectivity);
    return invert<T>(output);
}

// only works with integer images
template <typename T>
cv::Mat_<unsigned char> localMaxima(const cv::Mat& image, int connectivity) {
    CV_Assert(image.channels() == 1);

    // use morphologic reconstruction.
    cv::Mat marker = image - 1;
    cv::Mat_<unsigned char> candidates =
            marker < imreconstruct<T>(marker, image, connectivity);

    // now check the candidates
    // first pad the border
    cv::Scalar mn;
    T mx = std::numeric_limits<unsigned char>::max();
    cv::Mat_<unsigned char> output(candidates.size() + cv::Size(2,2));
    copyMakeBorder(candidates, output, 1, 1, 1, 1, cv::BORDER_CONSTANT, mx);
    cv::Mat input(image.size() + cv::Size(2,2), image.type());
    copyMakeBorder(image, input, 1, 1, 1, 1, cv::BORDER_CONSTANT, mn);

    int maxy = input.rows-1;
    int maxx = input.cols-1;
    int xminus, xplus;
    T val;
    T *iPtr, *iPtrMinus, *iPtrPlus;
    unsigned char *oPtr;
    cv::Rect reg(1, 1, image.cols, image.rows);
    cv::Scalar zero(0);
    cv::Scalar smx(mx);
    cv::Mat inputBlock = input(reg);

    // next iterate over image, and set candidates 
    //   that are non-max to 0 (via floodfill)
    for (int y = 1; y < maxy; ++y) {

        iPtr = input.ptr<T>(y);
        iPtrMinus = input.ptr<T>(y-1);
        iPtrPlus = input.ptr<T>(y+1);
        oPtr = output.ptr<unsigned char>(y);

        for (int x = 1; x < maxx; ++x) {

            // not a candidate, continue.
            if (oPtr[x] > 0) continue;

            xminus = x-1;
            xplus = x+1;

            val = iPtr[x];
            // compare values

            // 4 connected
            if ((val < iPtrMinus[x]) || (val < iPtrPlus[x]) 
                || (val < iPtr[xminus]) || (val < iPtr[xplus])) {

                // flood with type minimum value (only time when 
                //   the whole image may have mn is if it's flat)
                floodFill(inputBlock, output, cv::Point(xminus, y-1), smx, 
                    &reg, zero, zero, cv::FLOODFILL_FIXED_RANGE | 
                    cv::FLOODFILL_MASK_ONLY | connectivity);
                continue;
            }

            // 8 connected
            if (connectivity == 8) {
                if ((val < iPtrMinus[xminus]) || (val < iPtrMinus[xplus]) 
                    || (val < iPtrPlus[xminus]) || (val < iPtrPlus[xplus])) {

                    // flood with type minimum value (only time when 
                    //   the whole image may have mn is if it's flat)

                    floodFill(inputBlock, output, cv::Point(xminus, y-1), smx, 
                        &reg, zero, zero, cv::FLOODFILL_FIXED_RANGE | 
                        cv::FLOODFILL_MASK_ONLY | connectivity);
                    continue;
                }
            }
        }
    }
    return output(reg) == 0;  // similar to bitwise not.
}

template <typename T>
cv::Mat_<unsigned char> localMinima(const cv::Mat& image, int connectivity) {
    // only works for intensity images.
    CV_Assert(image.channels() == 1);

    cv::Mat cimage = invert<T>(image);
    return localMaxima<T>(cimage, connectivity);
}

// Operates on BINARY IMAGES ONLY
// perform bwlabel using union find.
cv::Mat_<int> bwlabel2(const cv::Mat& binaryImage, int connectivity, bool relab) {
    CV_Assert(binaryImage.channels() == 1);
    // only works for binary images.
    CV_Assert(binaryImage.type() == CV_8U);

    //copy, to make data continuous.
    cv::Mat input = cv::Mat::zeros(binaryImage.size(), binaryImage.type());
    binaryImage.copyTo(input);

    ConnComponents cc;
    cv::Mat_<int> output = cv::Mat_<int>::zeros(input.size());
    cc.label((unsigned char*) input.data, input.cols, input.rows, 
        (int *)output.data, -1, connectivity);

    input.release();

    return output;
}

// require padded image.
template <typename T>
cv::Mat border(cv::Mat& img, T background) {

    // SPECIFIC FOR OPEN CV CPU WATERSHED
    CV_Assert(img.channels() == 1);
    CV_Assert(std::numeric_limits<T>::is_integer);

    cv::Mat result(img.size(), img.type());
    T *ptr, *ptrm1, *res;

    for(int y=1; y< img.rows; y++){
        ptr = img.ptr<T>(y);
        ptrm1 = img.ptr<T>(y-1);

        res = result.ptr<T>(y);
        for (int x = 1; x < img.cols - 1; ++x) {
            if (ptrm1[x] == background && ptr[x] != background &&
                    ((ptrm1[x-1] != background && ptr[x-1] == background) ||
                (ptrm1[x+1] != background && ptr[x+1] == background))) {
                res[x] = background;
            } else {
                res[x] = ptr[x];
            }
        }
    }

    return result;
}

inline int *neighvector(int w, int conn) {
    static int neigh[8];
    int i = 0;
    switch(conn) {
        case 4:
            neigh[i++] = - w;
            neigh[i++] = - 1;
            neigh[i++] =   1;
            neigh[i++] =   w;
            break;
        case 8:
            neigh[i++] = - w - 1;
            neigh[i++] = - w;
            neigh[i++] = - w + 1;
            neigh[i++] = - 1;
            neigh[i++] =   1;
            neigh[i++] =   w - 1;
            neigh[i++] =   w;
            neigh[i++] =   w + 1;
            break;
        default:
            exit(1);
    }
    return neigh;
}

// input should have foreground > 0, and 0 for background
cv::Mat_<int> watershed(const cv::Mat& image, int connectivity) {
    // only works for intensity images.
    CV_Assert(image.channels() == 1);
    CV_Assert(image.type() ==  CV_16U);
    CV_Assert(connectivity == 4 || connectivity == 8);


    // create copy of the input image w/ border
    cv::Mat imageB;
    copyMakeBorder( image, imageB, 1, 1, 1, 1, cv::BORDER_CONSTANT, 
        std::numeric_limits<unsigned short int>::max());

    // create image to be labeled: same size as input image w/ border
    cv::Mat W(image.size(), CV_32S);
    W = UNVISITED;
    copyMakeBorder(W, W, 1, 1, 1, 1, cv::BORDER_CONSTANT, BORDER);

    CV_Assert(W.size() ==  imageB.size());

    // build neighbors vector
    int *neigh = neighvector(W.cols, connectivity);

    // TODO: reshape if not continuous
    CV_Assert(W.isContinuous() && imageB.isContinuous());

    // queue used for flooding
    std::list<int> Q;
    unsigned short int hmin, h;
    int qmin, q, p, i;
    int *WPtr = (int*)W.data;
    unsigned short int *imageBPtr = (unsigned short int*)imageB.data;

    // start in second row and stop before last row
    for(p = W.cols+1; p < W.cols*(W.rows-1); p++) {
        if(WPtr[p] == UNVISITED) {
            hmin = imageBPtr[p];
            qmin = p;
            for(i = 0; i < connectivity; i++) {
                q = p + neigh[i];
                h = imageBPtr[q];
                if(h < hmin) {
                    qmin = q;
                    hmin = h;
                }
            }
            if(imageBPtr[p] > hmin) {
                WPtr[p] = -qmin;
                Q.push_back(p);
            }
        }
    }
    while(!Q.empty()){
        p = Q.front();
        Q.pop_front();
        h = imageBPtr[p];
        for(i = 0; i < connectivity; i++) {
            q = p + neigh[i];
            // if it is not a peak or plateu
            if(WPtr[q] != UNVISITED || imageBPtr[q] != h) continue;
            WPtr[q] = -p;
            Q.push_back(q);
        }
    }
    int label, LABEL = 1, p1;

    // start in second row and stop before last row
    for(p = W.cols+1; p < W.cols*(W.rows-1); p++) {
        if(WPtr[p] == UNVISITED) {
            WPtr[p] = label = LABEL++;
            Q.clear();
            Q.push_back(p);
            h = imageBPtr[p];
            while(!Q.empty()) {
                p1 = Q.front();
                Q.pop_front();
                for(i = 0; i < connectivity; i++) {
                    q = p1 + neigh[i];
                    if(WPtr[q] == UNVISITED) {
                        WPtr[q] = label;
                        Q.push_back(q);
                    }
                }
            }

        }
    }
    // start in second row and stop before last row
    for(p = W.cols+1; p < W.cols*(W.rows-1); p++) {
        label = WPtr[p];
        if(label == BORDER || label > 0) continue;
        Q.clear();
        for(;;) {
            q = -label;
            label = WPtr[q];
            if(label > 0) break;
            Q.push_back(q);
        }
        WPtr[p] = label;
        while(!Q.empty()) {
            p1 = Q.front();
            Q.pop_front();
            WPtr[p1] = label;
        }

    }
    // remove border
    cv::Mat resultNoBorder;
    W(cv::Rect(1,1, image.cols, image.rows)).copyTo(resultNoBorder);

    return resultNoBorder;
}

// input should have foreground > 0, and 0 for background
cv::Mat_<int> watershed2(const cv::Mat& origImage, 
    const cv::Mat_<float>& image, int connectivity) {

    // only works for intensity images.
    CV_Assert(image.channels() == 1);
    CV_Assert(origImage.channels() == 3);

    cv::Mat minima = localMinima<float>(image, connectivity);
    cv::Mat_<int> labels = bwlabel2(minima, connectivity, true);

    // need borders, else get edges at edge.
    cv::Mat input, temp, output;
    copyMakeBorder(labels, temp, 1, 1, 1, 1, 
        cv::BORDER_CONSTANT, cv::Scalar_<int>(0));
    copyMakeBorder(origImage, input, 1, 1, 1, 1, 
        cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));


    // input: seeds are labeled from 1 to n, with 0 as background or 
    //   unknown regions output has -1 as borders.
    watershed(input, temp);
    output = border<int>(temp, (int)-1);

    return output(cv::Rect(1,1, image.cols, image.rows));
}

template <typename T>
cv::Mat morphOpen(const cv::Mat& image, const cv::Mat& kernel) {
    CV_Assert(kernel.rows == kernel.cols);
    CV_Assert(kernel.rows > 1);
    CV_Assert((kernel.rows % 2) == 1);

    int bw = (kernel.rows - 1) / 2;

    // can't use morphology Ex. the erode phase is not creating a border even 
    // though the method signature makes it appear that way. Because of this, 
    // and the fact that erode and dilate need different border values, have 
    // to do the erode and dilate myself. 
    cv::Mat t_image;

    copyMakeBorder(image, t_image, bw, bw, bw, bw, cv::BORDER_CONSTANT, 
        std::numeric_limits<unsigned char>::max());
    cv::Mat t_erode = cv::Mat::zeros(t_image.size(), t_image.type());
    erode(t_image, t_erode, kernel);

    cv::Mat erode_roi = t_erode(cv::Rect(bw, bw, image.cols, image.rows));
    cv::Mat t_erode2;
    copyMakeBorder(erode_roi,t_erode2, bw, bw, bw, bw, 
        cv::BORDER_CONSTANT, std::numeric_limits<unsigned char>::min());
    cv::Mat t_open = cv::Mat::zeros(t_erode2.size(), t_erode2.type());
    dilate(t_erode2, t_open, kernel);
    cv::Mat open = t_open(cv::Rect(bw, bw,image.cols, image.rows));

    t_open.release();
    t_erode2.release();
    erode_roi.release();
    t_erode.release();

    return open;
}

// templates specifications
template cv::Mat bwselect<unsigned char>(const cv::Mat& binaryImage, const cv::Mat& seeds, int connectivity);
template cv::Mat invert<unsigned char>(const cv::Mat &img);
template cv::Mat morphOpen<unsigned char>(const cv::Mat& image, const cv::Mat& kernel);
template cv::Mat imreconstruct<unsigned char>(const cv::Mat& seeds, const cv::Mat& image, int connectivity);
template cv::Mat imfillHoles<unsigned char>(const cv::Mat& image, bool binary, int connectivity);
template cv::Mat imhmin<float>(const cv::Mat& image, float h, int connectivity);

}
