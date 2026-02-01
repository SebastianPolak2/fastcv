import cv2
import torch
import fastcv

img = cv2.imread("/teamspace/studios/this_studio/fastcv/artifacts/test.jpg")
img_tensor = torch.from_numpy(img).cuda()
normalized_tensor = fastcv.normalize(img_tensor, 0.0, 255.0, 0.0, 255.0)
normalized_np = normalized_tensor.cpu().numpy()
normalized_np_cv2 = cv2.normalize(img, None, 0.0, 255.0, cv2.NORM_MINMAX)
cv2.imwrite("output_normalized.jpg", normalized_np.astype('uint8'))
cv2.imwrite("output_normalized_cv2.jpg", normalized_np_cv2.astype('uint8'))
print("saved normalized image.")