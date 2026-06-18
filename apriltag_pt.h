#pragma once

#include <stdint.h>

// A boundary point in the gradient clustering / quad fitting pipeline.
struct pt
{
    // Note: these represent 2*actual value.
    uint16_t x, y;
    int16_t gx, gy;
};

// a finished cluster: header and points in one allocation
struct pt_list
{
    int size;
    int pad; // keep pts 8-byte aligned
    struct pt pts[];
};
