#include <stdio.h>
#include <stdlib.h>

#include "apriltag.h"
#include "tagStandard52h13.h"
#include "common/image_u8.h"
#include "common/timeprofile.h"
#include "common/zarray.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s image.pgm nthreads\n", argv[0]);
        return 1;
    }
    image_u8_t *im = image_u8_create_from_pnm(argv[1]);
    if (im == NULL) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }

    apriltag_family_t *family = tagStandard52h13_create();
    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family(td, family);
    td->nthreads = atoi(argv[2]);
    td->quad_decimate = 1.0f;
    td->quad_sigma = 0.0f;
    td->refine_edges = 1;
    td->decode_sharpening = 0.0;

    for (int i = 0; i < 5; i++) {
        zarray_t *warmup = apriltag_detector_detect(td, im);
        apriltag_detections_destroy(warmup);
    }

    zarray_t *detections = apriltag_detector_detect(td, im);
    printf("detections: %d\n", zarray_size(detections));
    timeprofile_display(td->tp);

    apriltag_detections_destroy(detections);
    apriltag_detector_destroy(td);
    tagStandard52h13_destroy(family);
    image_u8_destroy(im);
    return 0;
}
