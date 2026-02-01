import time

import cv2
import torch
import fastcv
import numpy as np

def benchmark_normalize(sizes = [(512, 512), (1024, 1024), (2048, 2048)], iterations=10):
    results = []
    for size in sizes:
        height, width = size
        img = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        img_tensor = torch.from_numpy(img).cuda()

        for _ in range(5):
            _ = fastcv.normalize(img_tensor, 0.0, 255.0, 0.0, 255.0)

        # Benchmark fastcv
        start_time = time.time()
        for _ in range(iterations):
            _ = fastcv.normalize(img_tensor, 0.0, 255.0, 0.0, 255.0)
        end_time = time.time()

        fc_time = ((end_time - start_time) / iterations) * 1000  # ms

        # Benchmark opencv
        img_cv = img.astype(np.float32)
        start_time = time.time()
        for _ in range(iterations):
            _ = cv2.normalize(img_cv, None, 0.0, 255.0, cv2.NORM_MINMAX)
        end_time = time.time()

        cv_time = ((end_time - start_time) / iterations) * 1000  # ms

        results.append((size, cv_time, fc_time))
        print(f"Size: {size}, fastcv Time: {fc_time:.4f} ms, OpenCV Time: {cv_time:.4f} ms")

    return results

if __name__ == "__main__":
    results = benchmark_normalize()
    print("\n=== Benchmark Results ===")
    print("Size\t\tOpenCV (CPU)\tfastcv (CUDA)")
    for size, cv_time, fc_time in results:
        print(f"{size}x{size}\t{cv_time:.4f} ms\t{fc_time:.4f} ms")