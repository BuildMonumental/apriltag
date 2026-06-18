import sys
import time
from pathlib import Path

from terra.devices.galaxy import open_camera

serial, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
cam = open_camera(is_gige=False, serial_number=serial)
for _ in range(10):
    cam.get_view()
views = []
t0 = time.time()
while time.time() - t0 < 10 and len(views) < 200:
    views.append(cam.get_view())
dt = time.time() - t0
print(f"captured {len(views)} frames in {dt:.1f}s ({len(views)/dt:.1f} fps), saving...")
for i, v in enumerate(views):
    v.save(out / f"burst_{i:04d}")
print("saved")
