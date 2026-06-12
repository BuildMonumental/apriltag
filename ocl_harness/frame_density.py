import sys

import cv2
import numpy as np

for path in sys.argv[1:]:
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = im.shape
    tw, th = w // 4, h // 4
    crop = im[:th * 4, :tw * 4]
    tiles = crop.reshape(th, 4, tw, 4)
    tmin, tmax = tiles.min(axis=(1, 3)), tiles.max(axis=(1, 3))
    k = np.ones((3, 3), np.uint8)
    bmin, bmax = cv2.erode(tmin, k), cv2.dilate(tmax, k)
    mn = np.repeat(np.repeat(bmin, 4, 0), 4, 1).astype(int)
    mx = np.repeat(np.repeat(bmax, 4, 0), 4, 1).astype(int)
    grey = (mx - mn) < 5
    t = np.where(grey, 127, np.where(crop > mn + (mx - mn) // 2, 255, 0))
    pairs = (((t[:, :-1] + t[:, 1:]) == 255).sum() +
             ((t[:-1, :] + t[1:, :]) == 255).sum())
    active = (t != 127).mean()
    name = path.split("/")[3]
    print(f"{name}: boundary pairs {pairs/1e6:.2f}M, active px {active*100:.0f}%")
