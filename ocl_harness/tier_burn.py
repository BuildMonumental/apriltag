import glob
import os
import resource
import statistics
import threading
import time

import cv2

os.environ["OCL_ICD_VENDORS"] = "/run/opengl-driver/etc/OpenCL/vendors"
from dt_apriltags import Detector

img = cv2.imread("/tmp/probeframe.png", cv2.IMREAD_GRAYSCALE)
det = Detector(families="tagStandard52h13", nthreads=4, quad_decimate=1.0,
               searchpath=["/var/lib/vide-gputest"])

tempPaths = glob.glob("/sys/class/thermal/thermal_zone*/temp")
freqPath = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
trace = []
phase = ["startup"]
stop = [False]


def sampler():
    while not stop[0]:
        time.sleep(5)
        trace.append((phase[0],
                      max(int(open(p).read()) for p in tempPaths) // 1000,
                      int(open(freqPath).read()) // 1000))


def drmBusyNs():
    total = 0
    for f in glob.glob("/proc/self/fdinfo/*"):
        try:
            for line in open(f):
                if line.startswith("drm-engine-compute"):
                    total = max(total, int(line.split()[1]))
        except OSError:
            pass
    return total


threading.Thread(target=sampler, daemon=True).start()

PHASES = [("fullGpu", {"APRILTAG_OPENCL": "1", "APRILTAG_OPENCL_FIT": "1"}, 60),
          ("midway", {"APRILTAG_OPENCL": "1"}, 60),
          ("stockCpu", {}, 300)]

for name, env, seconds in PHASES:
    for k in ("APRILTAG_OPENCL", "APRILTAG_OPENCL_FIT"):
        os.environ.pop(k, None)
    os.environ.update(env)
    det.detect(img)
    det.detect(img)
    phase[0] = name
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    c0 = r0.ru_utime + r0.ru_stime
    g0 = drmBusyNs()
    times = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        s = time.perf_counter()
        n = len(det.detect(img))
        times.append((time.time(), (time.perf_counter() - s) * 1000))
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    coreMs = (r1.ru_utime + r1.ru_stime - c0) * 1000 / len(times)
    gpuMs = (drmBusyNs() - g0) / 1e6 / len(times)
    mid = t0 + seconds / 2
    early = statistics.median(ms for ts, ms in times if ts < t0 + 30)
    late = statistics.median(ms for ts, ms in times if ts > max(mid, t0 + seconds - 30))
    overall = statistics.median(ms for _, ms in times)
    phaseTrace = [(t, f) for p, t, f in trace if p == name]
    tFirst, fFirst = phaseTrace[0] if phaseTrace else (0, 0)
    tLast, fLast = phaseTrace[-1] if phaseTrace else (0, 0)
    print(f"{name:9s} {len(times):4d} detects ({n} tags) | wall med {overall:5.0f} ms "
          f"(early {early:5.0f} / late {late:5.0f}) | {coreMs:5.0f} core-ms | {gpuMs:5.1f} GPU-ms | "
          f"temp {tFirst}->{tLast}C cpu0 {fFirst}->{fLast}MHz", flush=True)

stop[0] = True
