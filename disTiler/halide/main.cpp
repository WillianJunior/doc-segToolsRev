#include <iostream>
#include <ctime>
#include <string>

#include "Halide.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "BlurJIT.h"

// break input into bg and dense:
//  get hotspots through threshold
//      * can use user-defined functions
//      dilate
//  crop dense into rectangles
//  make bg rectangles
//      * merge bg's if merged is rectangular (this is done to reduce borders count)
// break dense list to reach 2*np dense tiles:
//  each dense is an initial 1 node kd-tree
//  make priority queue on the leaf nodes of all kd-trees ordered by cost
//      * use smart queue structures (fibonacci tree?)
//  until number of leaf nodes < 2*np:
//      split leaf node with highest cost
//      add new leaf nodes
//      re-sort queue
// make borders list:
//  for each rectangle r in bg and dense
//      insert 4 borders on a map
// distributed execution:
//  for each dense
//      send and execute
//  for each border
//      send and execute
//  for each bg
//      send and execute
// * distributed execution with border locality
//  for each pair of leafs on the dense kd-tree
//      send and execute 2 dense on 2 nodes
//      on each execute finish, send results to root node on a daemon
//      on slack, calculate bg tiles
//      on both finish, redistribute border area between themselves
//      execute half border on both nodes
//      merge on next level to calculate border
//      on last level border finish, return borders to root node
//  * can use root node as manager, which only performs transfers and calculate bg on slack

using namespace std;

int find_arg_pos(string s, int argc, const char** argv) {
    for (int i=1; i<argc; i++)
        if (string(argv[i]).compare(s)==0)
            return i;
    return -1;
}

int main(int argc, char const *argv[]) {
    
    if (argc < 2) {
        cout << "usage: tilerTest -d <N> -i <input image>" << endl;
        cout << "\t-d: 0 -> serial execution" << endl;
        cout << "\t    1 -> full distributed execution" << endl;
        return 0;
    }

    // Distribution level
    enum Paral_t {p_serial, p_dist};
    Paral_t paral;
    if (find_arg_pos("-d", argc, argv) == -1) {
        cout << "Missing distribution level." << endl;
        return 0;
    } else
        paral = static_cast<Paral_t>(atoi(argv[find_arg_pos("-d", argc, argv)+1]));

    // Input image
    string img_path;
    if (find_arg_pos("-i", argc, argv) == -1) {
        cout << "Missing input image." << endl;
        return 0;
    } else
        img_path = argv[find_arg_pos("-i", argc, argv)+1];
    cv::Mat input = cv::imread(img_path, cv::IMREAD_GRAYSCALE);

    cv::Mat output = cv::Mat::zeros(input.size(), input.type());

    switch (paral) {
        case p_serial:
            BlurJIT blur(input, output);
            blur.sched();
            blur.run();

            break;
        case p_dist:
            // break tiles
            list<tiles_t> tilesRects;

            // perform distributed execution
            distExec();
            
            

    };

    cv::imwrite("./input.png", input);
    cv::imwrite("./output.png", output);
    return 0;
}