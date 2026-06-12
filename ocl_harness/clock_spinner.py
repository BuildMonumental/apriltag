import time

windows = []
end = time.time() + 50
while time.time() < end:
    t0 = time.perf_counter()
    n = 0
    x = 1.0
    while time.perf_counter() - t0 < 1.0:
        for _ in range(10000):
            x = x * 1.0000001 + 0.1
        n += 10000
    windows.append(n / 1e6)
    print(f"{time.strftime('%H:%M:%S')} {windows[-1]:.2f} Miter/s", flush=True)
