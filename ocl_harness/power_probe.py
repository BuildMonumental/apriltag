import glob
import os
import statistics
import subprocess
import sys
import threading
import time

import cv2

os.environ["OCL_ICD_VENDORS"] = "/run/opengl-driver/etc/OpenCL/vendors"
from dt_apriltags import Detector

PHASE_SECONDS = 15
cpuDirs = sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*"), key=lambda p: int(p.split("cpu")[-1]))
maxFreqs = [int(open(c + "/cpufreq/cpuinfo_max_freq").read()) for c in cpuDirs]
pMax = max(maxFreqs)
pCores = [c for c, m in zip(cpuDirs, maxFreqs) if m == pMax]
raplPath = next(iter(glob.glob("/sys/class/powercap/intel-rapl:0/energy_uj")), None)
gpuPath = next(iter(glob.glob("/sys/class/drm/card0/gt_cur_freq_mhz")), None)

samples = []
phase = ["startup"]
stop = [False]


def sampler():
    lastE = int(open(raplPath).read()) if raplPath else 0
    lastT = time.time()
    while not stop[0]:
        time.sleep(0.25)
        now = time.time()
        e = int(open(raplPath).read()) if raplPath else 0
        watts = (e - lastE) / 1e6 / (now - lastT) if raplPath and e >= lastE else None
        lastE, lastT = e, now
        samples.append({
            "phase": phase[0],
            "spinCore": int(open(pCores[0] + "/cpufreq/scaling_cur_freq").read()) // 1000,
            "pCores": [int(open(c + "/cpufreq/scaling_cur_freq").read()) // 1000 for c in pCores],
            "gpu": int(open(gpuPath).read()) if gpuPath else None,
            "watts": watts,
        })


img = cv2.imread("/tmp/probeframe.png", cv2.IMREAD_GRAYSCALE)
det = Detector(families="tagStandard52h13", nthreads=4, quad_decimate=1.0,
               searchpath=["/var/lib/vide-gputest"])
os.environ.pop("APRILTAG_OPENCL", None)
det.detect(img)

spinSrc = "import time\nx = 1.0\nwhile True:\n    for _ in range(100000): x = x * 1.0000001 + 0.1\n"

t = threading.Thread(target=sampler, daemon=True)
t.start()

detectTimes = {}


def runPhase(name, spin, detectMode):
    spinner = None
    if spin:
        spinner = subprocess.Popen(["taskset", "-c", pCores[0].split("cpu")[-1],
                                    sys.executable, "-c", spinSrc],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)
    phase[0] = name
    end = time.time() + PHASE_SECONDS
    times = []
    if detectMode == "cpu":
        os.environ.pop("APRILTAG_OPENCL", None)
        os.environ.pop("APRILTAG_OPENCL_FIT", None)
    elif detectMode == "gpu":
        os.environ["APRILTAG_OPENCL"] = "1"
        os.environ["APRILTAG_OPENCL_FIT"] = "1"
    if detectMode:
        det.detect(img)
    while time.time() < end:
        if detectMode:
            s = time.perf_counter()
            det.detect(img)
            times.append((time.perf_counter() - s) * 1000)
        else:
            time.sleep(0.2)
    phase[0] = "between"
    if times:
        detectTimes[name] = statistics.median(times)
    if spinner:
        spinner.terminate()
    time.sleep(1)


runPhase("idle", spin=False, detectMode=None)
runPhase("spinAlone", spin=True, detectMode=None)
runPhase("spinPlusCpuDetect", spin=True, detectMode="cpu")
runPhase("spinPlusGpuDetect", spin=True, detectMode="gpu")
stop[0] = True
t.join(timeout=2)

print(f"P-cores: {len(pCores)} threads @ max {pMax//1000} MHz nominal")
for name in ("idle", "spinAlone", "spinPlusCpuDetect", "spinPlusGpuDetect"):
    rows = [s for s in samples if s["phase"] == name]
    spinF = statistics.median(r["spinCore"] for r in rows)
    allP = statistics.median(f for r in rows for f in r["pCores"])
    gpuF = statistics.median(r["gpu"] for r in rows) if rows[0]["gpu"] is not None else 0
    w = [r["watts"] for r in rows if r["watts"]]
    watts = statistics.median(w) if w else 0
    dt = detectTimes.get(name)
    extra = f", detect {dt:.0f} ms" if dt else ""
    print(f"{name:20s} spinCore {spinF:4.0f} MHz | P-median {allP:4.0f} MHz | GPU {gpuF:4.0f} MHz | pkg {watts:5.1f} W{extra}")
