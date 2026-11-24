import time

import cv2
import torch
import fastcv
import numpy as np

def benchmark_normalize(sizes=[1024, 2048, 4096], runs=50):
    results = []

    for size in sizes:
        print(f"\n Benchmarking {size}x{size}x3 image")

        img_np = np.random.randint(60, 180, (size, size, 3), dtype=np.uint8) # niski kontrast
        img_torch = torch.from_numpy(img_np).cuda()

        start = time.perf_counter()
        for _ in range(runs):
            _ = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX)
        end = time.perf_counter()
        cv_time = (end - start) / runs * 1000

        for _ in range(5):  # warm-up
            _ = fastcv.normalize(img_torch, 0, 255)

        torch.cuda.synchronize()  # czekanie na koniec warm-up
        start = time.perf_counter()
        for _ in range(runs):
            _ = fastcv.normalize(img_torch, 0, 255)
        torch.cuda.synchronize()  # czekanie na koniec obliczen GPU
        end = time.perf_counter()
        fc_time = (end - start) / runs * 1000

        # z uwzgl kopiowaniem
        start = time.perf_counter()
        for _ in range(runs):
            _ = fastcv.normalize(torch.from_numpy(img_np).cuda(), 0, 255).cpu().numpy()
        end = time.perf_counter()
        e2e_time = (end - start) / runs * 1000


        results.append((size, cv_time, fc_time, e2e_time))
        print(f"OpenCV (CPU): {cv_time:.4f} ms  |  fastcv (CUDA): {fc_time:.4f} ms  |  fastcv + copies: {e2e_time:.4f} ms")

    return results


if __name__ == "__main__":
    results = benchmark_normalize()

    print("\n RESULTS")
    print("Size\t\tOpenCV (CPU)\tfastcv (CUDA)\tSpeedup\tfastcv + copies\tSpeedup")
    for size, cv_time, fc_time, e2e_time in results:
        print(f"{size}x{size}\t{cv_time:.4f} ms\t{fc_time:.4f} ms\t{cv_time / fc_time:.2f}x" f"\t{e2e_time:.4f} ms\t{cv_time / e2e_time:.2f}x")
