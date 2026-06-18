import os
import resource
import statistics
import time

import cv2
import numpy as np
from dt_apriltags import Detector

os.environ["APRILTAG_OPENCL"] = "1"
os.environ["APRILTAG_OPENCL_FIT"] = "1"

image = cv2.imread("/tmp/raw_color.pgm", cv2.IMREAD_GRAYSCALE)
print(f"fixture: {image.shape[1]}x{image.shape[0]}")


def makeDetector(searchpath):
    kwargs = dict(families="tagStandard52h13", nthreads=4, quad_decimate=1.0,
                  quad_sigma=0.0, refine_edges=1, decode_sharpening=0.0)
    if searchpath is not None:
        kwargs["searchpath"] = searchpath
    try:
        return Detector(min_cluster_pixels=100, **kwargs)
    except TypeError:
        return Detector(**kwargs)


cpuDet = makeDetector(None)
gpuDet = makeDetector(["/tmp/atspike/build"])

cpuRes = sorted(cpuDet.detect(image), key=lambda d: d.tag_id)
gpuRes = sorted(gpuDet.detect(image), key=lambda d: d.tag_id)
print(f"tags: stock {len(cpuRes)}  gpu {len(gpuRes)}")
ids = [d.tag_id for d in cpuRes] == [d.tag_id for d in gpuRes]
maxDelta = max(float(np.abs(c.corners - g.corners).max()) for c, g in zip(cpuRes, gpuRes))
print(f"id match: {ids}  max corner delta: {maxDelta:.9f} px")


def benchLoop(det, iters=10):
    times = []
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    c0 = r0.ru_utime + r0.ru_stime
    for _ in range(iters):
        start = time.perf_counter()
        det.detect(image)
        times.append((time.perf_counter() - start) * 1000)
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    coreMs = (r1.ru_utime + r1.ru_stime - c0) * 1000 / iters
    return statistics.median(times), min(times), coreMs


for det in (cpuDet, gpuDet):
    benchLoop(det, 3)

print(f"{'round':>5} {'lib':>5} {'median_ms':>10} {'min_ms':>8} {'core_ms':>8}")
for rnd in range(3):
    for name, det in (("stock", cpuDet), ("gpu", gpuDet)):
        med, mn, core = benchLoop(det)
        print(f"{rnd:>5} {name:>5} {med:>10.1f} {mn:>8.1f} {core:>8.1f}")
