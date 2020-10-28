#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <limits.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    int i, c, x, y, kx, ky;
    int koffset = (l.size - 1) / 2;
    
    // TODO: 6.1 - iterate over the input and fill in the output with max values
    for (i = 0; i < in.rows; i++) {
        for(c = 0; c < l.channels; c++) {
        // for each top-left corner of the convolution grid
        // i.e. every single element 
            for(y = 0; y < l.height; y += l.stride) {
                for(x = 0; x < l.width; x += l.stride) {
                // x, y on image = i, j is the center of the kernel
                // for each row in the convolution grid 
                    // so from size / 2;
                    // for each column in the convolution grid 
                    float max = FLT_MIN; 
                    for(ky = 0; ky < l.size; ky++) {
                        int imgy = y + ky - koffset;
                        if (imgy < 0 || imgy >= l.height) {
                            continue;
                        }
                        for(kx = 0; kx < l.size; kx++) {
                            int imgx = x + kx - koffset;
                            if (imgx < 0 || imgx >= l.width) {
                                continue;
                            }
                            // for every x we need to offset by 1
                            // for every y we need to offset by a full row l.width
                            // for every c we need to offset by a full single-channel image l.width * l.height
                            // for every multi-channel image (a row) we need to offset by l.width * l.height * l.channels
                            // so imgx + l.width * imgy + l.width * l.height * c + l.width * l.height * l.channels * i
                            // Which we factor to get this
                            float val = in.data[imgx + l.width * (imgy + l.height * (c + l.channels * i))];
                            
                            if (val > max) {
                                max = val;
                            }
                        }
                    }
                    // Update max here
                    int outx = (x / l.stride);
                    int outy = (y / l.stride);

                    out.data[outx + outw * (outy + outh * (c + l.channels * i))] = max;
                }
            }
        }
    }

    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    matrix in    = *l.x;
    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
       int i, c, x, y, kx, ky;
    int koffset = (l.size - 1) / 2;
    
    // TODO: 6.1 - iterate over the input and fill in the output with max values
    for (i = 0; i < in.rows; i++) {
        for(c = 0; c < l.channels; c++) {
        // for each top-left corner of the convolution grid
        // i.e. every single element 
            for(y = 0; y < l.height; y += l.stride) {
                for(x = 0; x < l.width; x += l.stride) {
                // x, y on image = i, j is the center of the kernel
                // for each row in the convolution grid 
                    // so from size / 2;
                    // for each column in the convolution grid 
                    float max = FLT_MIN; 
                    int mx = 0;
                    int my = 0;
                    for(ky = 0; ky < l.size; ky++) {
                        int imgy = y + ky - koffset;
                        if (imgy < 0 || imgy >= l.height) {
                            continue;
                        }
                        for(kx = 0; kx < l.size; kx++) {
                            int imgx = x + kx - koffset;
                            if (imgx < 0 || imgx >= l.width) {
                                continue;
                            }
                            // for every x we need to offset by 1
                            // for every y we need to offset by a full row l.width
                            // for every c we need to offset by a full single-channel image l.width * l.height
                            // for every multi-channel image (a row) we need to offset by l.width * l.height * l.channels
                            // so imgx + l.width * imgy + l.width * l.height * c + l.width * l.height * l.channels * i
                            // Which we factor to get this
                            float val = in.data[imgx + l.width * (imgy + l.height * (c + l.channels * i))];
                            
                            if (val > max) {
                                max = val;
                                mx = imgx;
                                my = imgy;
                            }
                        }
                    }
                    // Update max here
                    int outx = (x / l.stride);
                    int outy = (y / l.stride);
                    //
                    dx.data[mx + l.width * (my + l.height * (c + l.channels * i))] += dy.data[outx + outw * (outy + outh * (c + l.channels * i))];
                }
            }
        }
    }


    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

