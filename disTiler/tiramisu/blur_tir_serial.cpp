#include <Halide.h>
#include "../include/tiramisu/core.h"
#include "blur_tir.h"

// using namespace tiramisu;

int main(int argc, char **argv) {

    tiramisu::init();

     // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    tiramisu::function function0("blur_tir_serial");
    // tiramisu::constant N("N", expr((int32_t) size), p_int32, true, NULL, 0, &function0);
    tiramisu::var N = var("N");
    tiramisu::var i = var("i");
    tiramisu::computation input("[N]->{input[i]}", expr(), false, p_uint8, &function0);
    tiramisu::computation result_init("[N]->{result_init[0]}", expr(input(0)), true, p_uint8, &function0);
    tiramisu::computation result("[N]->{result[i]: 1<=i<N}", expr(), true, p_uint8, &function0);
    result.set_expression((result(i - 1) + input(i)));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    result.after(result_init, computation::root);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer input_buffer("input_buffer", {size}, p_uint8, a_input, &function0);
    buffer input_size("input_size", {1}, p_uint8, a_input, &function0);
    buffer result_scalar("result_scalar", {1}, p_uint8, a_output, &function0);
    input.set_access("[N]->{input[i]->input_buffer[i]}");
    result_init.set_access("[N]->{result_init[i]->result_scalar[0]}");
    result.set_access("[N]->{result[i]->result_scalar[0]}");

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.codegen({&input_buffer, &input_size, &result_scalar}, "build/generated_fct_developers_tutorial_05.o");

}