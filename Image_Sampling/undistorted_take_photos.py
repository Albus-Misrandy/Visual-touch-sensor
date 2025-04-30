"""
Author: Albus.Misrandy
Function: 拍摄矫正后的双目图像（保存到同一文件夹）
"""

import cv2
import os
import numpy as np

# ==== 标定参数 ====
mtx_l = np.array([[333.65230404, 0, 324.21396554],
                  [0, 336.00751069, 249.13689605],
                  [0, 0, 1]])
dist_l = np.array([-0.04989725, 0.24822917, 0.0030429, 0.00444644, -0.2585821])

mtx_r = np.array([[328.51416857, 0, 300.69223395],
                  [0, 331.11410595, 260.83849052],
                  [0, 0, 1]])
dist_r = np.array([3.25171573e-04, 8.30036337e-02, -3.86796976e-05, -1.14187376e-03, -1.08096254e-01])

# ==== 保存路径（统一目录）====
save_dir = 'captured_images'
os.makedirs(save_dir, exist_ok=True)

# ==== 打开摄像头 ====
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

if not cap_left.isOpened() or not cap_right.isOpened():
    if not cap_left.isOpened():
        print("sfds")
    if not cap_right.isOpened():
        print("fuck")

# ==== 获取一帧图像，计算映射 ====
ret1, frame_left = cap_left.read()
ret2, frame_right = cap_right.read()

h_l, w_l = frame_left.shape[:2]
h_r, w_r = frame_right.shape[:2]

map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, None, mtx_l, (w_l, h_l), cv2.CV_16SC2)
map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, None, mtx_r, (w_r, h_r), cv2.CV_16SC2)

print("映射表已生成，按 'c' 拍照，按 'ESC' 退出")

count = 0
while True:
    ret1, frame_l = cap_left.read()
    ret2, frame_r = cap_right.read()

    if not ret1 or not ret2:
        print("图像捕获失败")
        break

    undistorted_l = cv2.remap(frame_l, map1_l, map2_l, interpolation=cv2.INTER_LINEAR)
    undistorted_r = cv2.remap(frame_r, map1_r, map2_r, interpolation=cv2.INTER_LINEAR)

    # 实时预览
    cv2.imshow("Left Undistorted", undistorted_l)
    cv2.imshow("Right Undistorted", undistorted_r)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        img_name_l = os.path.join(save_dir, f"{count:04d}.jpg")
        img_name_r = os.path.join(save_dir, f"{count+1:04d}.jpg")
        cv2.imwrite(img_name_l, undistorted_l)
        cv2.imwrite(img_name_r, undistorted_r)
        count += 2
        print(f"[✔] 第 {count} 对图像已保存: {img_name_l}, {img_name_r}")

    elif key == 27:
        print("退出程序")
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
