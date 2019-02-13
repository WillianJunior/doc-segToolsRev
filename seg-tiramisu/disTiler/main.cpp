#include <iostream>
#include <ctime>

#include "Halide.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "pixelsOps.hpp"

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

int find_arg_pos(string s, int argc, char** argv) {
    for (int i=1; i<argc; i++)
        if (string(argv[i]).compare(s)==0)
            return i;
    return -1;
}

int main(int argc, char const *argv[]) {
    
    if (argc != 2) {
        cout << "usage: tilerTest <>"
        return 0;
    }

    enum Paral_t {serial, dist};
    Paral_t paral = Paral_t.serial;
    if (find_arg_pos("-d", argc, argv) != -1)
        paral = Paral_t.dist;


    switch (paral) {
        case Paral_t.dist:
            int rank = tiramisu_MPI_init()

            // Allocate buffers for each rank and fill as appropriate
            Halide::Buffer<uint32_t> input(_COLS + 2, _ROWS/10 + 2, "input");

            // perform computation
            MPI_Barrier(MPI_COMM_WORLD);
            blurxy(input.raw_buffer(), output.raw_buffer());
            MPI_Barrier(MPI_COMM_WORLD);

            // Need to cleanup MPI appropriately
            tiramisu_MPI_cleanup();
            break;
    }

    return 0;
}