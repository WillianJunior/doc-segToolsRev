#include "Halide.h"

// Generators are a more structured way to do ahead-of-time
// compilation of Halide pipelines. Instead of writing an int main()
// with an ad-hoc command-line interface like we did in lesson 10, we
// define a class that inherits from Halide::Generator.
class MyFirstGenerator : public Halide::Generator<MyFirstGenerator> {
public:
    // We declare the Inputs to the Halide pipeline as public
    // member variables. They'll appear in the signature of our generated
    // function in the same order as we declare them.
    Input<uint8_t> offset{"offset"};
    Input<Buffer<uint8_t>> input{"input", 2};

    // We also declare the Outputs as public member variables.
    Output<Buffer<uint8_t>> brighter{"brighter", 2};

    // Typically you declare your Vars at this scope as well, so that
    // they can be used in any helper methods you add later.
    Var x, y;

    // We then define a method that constructs and return the Halide
    // pipeline:
    void generate() {
        // In lesson 10, here is where we called
        // Func::compile_to_file. In a Generator, we just need to
        // define the Output(s) representing the output of the pipeline.
        brighter(x, y) = input(x, y) + offset;

        // Schedule it.
        brighter.vectorize(x, 16).parallel(y);
    }
};

// We compile this file along with tools/GenGen.cpp. That file defines
// an "int main(...)" that provides the command-line interface to use
// your generator class. We need to tell that code about our
// generator. We do this like so:
HALIDE_REGISTER_GENERATOR(MyFirstGenerator, my_first_generator)