import cv2
import torch
import fastcv

img = cv2.imread("artifacts/test.jpg")
low = cv2.convertScaleAbs(img, alpha=0.3, beta=90)

img_tensor = torch.from_numpy(low).cuda()
normalized_tensor = fastcv.normalize(img_tensor, 0, 255)
normalized_np = normalized_tensor.cpu().numpy()

normalized_cv2 = cv2.normalize(low, None, 0, 255, cv2.NORM_MINMAX)

print(f"input range: {low.min()}..{low.max()}, output range: {normalized_np.min()}..{normalized_np.max()}")
cv2.imwrite("output_low_contrast.jpg", low)
cv2.imwrite("output_normalized.jpg", normalized_np)
cv2.imwrite("output_normalized_cv2.jpg", normalized_cv2)

