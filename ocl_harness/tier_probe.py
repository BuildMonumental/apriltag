import os
import resource
import statistics
import sys
import time

import cv2
import numpy as np

os.environ["OCL_ICD_VENDORS"] = "/run/opengl-driver/etc/OpenCL/vendors"

from dt_apriltags import Detector

image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
print(f"frame: {image.shape[1]}x{image.shape[0]}")


def make(searchpath):
    kwargs = dict(families="tagStandard52h13", nthreads=4, quad_decimate=1.0,
                  quad_sigma=0.0, refine_edges=1, decode_sharpening=0.0)
    if searchpath:
        kwargs["searchpath"] = searchpath
    try:
        return Detector(min_cluster_pixels=100, **kwargs)
    except TypeError:
        return Detector(**kwargs)


stock = make(None)
gpu = make(["/tmp"])

tiers = [
    ("stock", stock, {}),
    ("gpu-frontend", gpu, {"APRILTAG_OPENCL": "1"}),
    ("gpu-fit", gpu, {"APRILTAG_OPENCL": "1", "APRILTAG_OPENCL_FIT": "1"}),
]

results = {}
for name, det, env in tiers:
    for k in ("APRILTAG_OPENCL", "APRILTAG_OPENCL_FIT"):
        os.environ.pop(k, None)
    os.environ.update(env)
    det.detect(image)
    det.detect(image)
    times = []
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    c0 = r0.ru_utime + r0.ru_stime
    for _ in range(5):
        t0 = time.perf_counter()
        res = det.detect(image)
        times.append((time.perf_counter() - t0) * 1000)
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    core = (r1.ru_utime + r1.ru_stime - c0) * 1000 / 5
    results[name] = sorted(res, key=lambda d: d.tag_id)
    print(f"{name:14s} {len(res)} tags, median {statistics.median(times):6.0f} ms wall, {core:5.0f} core-ms")

ref = results["stock"]
for name in ("gpu-frontend", "gpu-fit"):
    r = results[name]
    same = len(ref) == len(r) and all(a.tag_id == b.tag_id and np.array_equal(a.corners, b.corners) for a, b in zip(ref, r))
    print(f"parity {name}: identical={same}")
