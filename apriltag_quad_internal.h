/* Internal structures shared between the quad-threshold pipeline and the
 * OpenCL acceleration module. Not installed; not part of the public API. */
#pragma once

#include <stdint.h>

// a maximal horizontal segment of equal non-127 pixels, x in [0, w-2]
// (the last column never participates in runs; it is only reachable as a
// diagonal neighbor of a white run ending at w-2)
struct row_run
{
    uint16_t start, end; // inclusive
    uint8_t v;
};

// one boundary point of a gradient cluster
struct pt
{
    // Note: these represent 2*actual value.
    uint16_t x, y;
    int16_t gx, gy;
};
