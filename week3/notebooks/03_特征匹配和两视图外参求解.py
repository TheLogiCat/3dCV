import numpy as np
import cv2

imgpath1 = '../data/images/aloeL.jpg'
imgpath2 = '../data/images/aloeR.jpg'

img1 = cv2.imread(imgpath1)
img2 = cv2.imread(imgpath2)

img1 = cv2.resize(img1, (400, 600))
img2 = cv2.resize(img2, (400, 600))

if img1 is None or img2 is None:
    print("无法读取图片，请检查路径。")

# 定义一个近似的内参矩阵 (对于 Aloe 数据集，这是一个估计值)
# Aloe 图像尺寸通常是 1282x1110
h, w = img1.shape[:2]
focal_length = w * 1.2  # 估算焦距
K = np.array([[focal_length, 0, w/2],
              [0, focal_length, h/2],
              [0, 0, 1]], dtype=np.float64)

# --- 接下来直接运行特征匹配逻辑 ---
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

# 计算本质矩阵
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
print(f"\n本质矩阵 E:\n{E}")

# 恢复姿态
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
print(f"\n计算出的位移 t (应该主要是 x 轴方向的移动):\n{t}")

# 可视化
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good[:50], None, flags=2) # 只画前50个
cv2.imshow("Matches", img_matches) # 如果在本地运行可取消注释
cv2.waitKey(0)