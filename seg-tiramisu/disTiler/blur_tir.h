#ifndef BLUR_TIR_H
#define BLUR_TIR_H

#include <tiramisu/utils.h>

// temporary size of images definition
#define M 100
#define N 100

#ifdef __cplusplus
extern "C" {
#endif

int blur_tir_serial(halide_buffer_t *_p0_buffer, halide_buffer_t *_p1_buffer);
int blur_tir_dist(halide_buffer_t *_p0_buffer, halide_buffer_t *_p1_buffer);
// int blurxy_argv(void **args); // don't know what this is for...

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif //BLUR_TIR_H