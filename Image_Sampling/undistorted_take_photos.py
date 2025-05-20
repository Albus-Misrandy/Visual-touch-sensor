"""
Author: Albus.Misrandy
Function: 拍摄矫正后的双目图像（保存到同一文件夹）
"""

import cv2
import os
import numpy as np

# ==== 标定参数 ====
mtx_l = np.array([[377.62484394, 0, 338.22935123],
                  [0, 381.64291743, 248.49561413],
                  [0, 0, 1]])
dist_l = np.array([-0.25887591, 0.00739472, 0.00184546, -0.00079176, 0.02526219])

mtx_r = np.array([[372.31843353, 0, 343.82837719],
                  [0, 375.63300679, 253.46532185],
                  [0, 0, 1]])
dist_r = np.array([-0.2614851, 0.06556446, -0.0011021, -0.00212242, -0.0439001])

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

R1 = np.array([[0.99902054, -0.03584055, -0.02595043],
              [0.03578368, 0.99935604, -0.00265274],
              [0.02602879, 0.00172154, 0.99965971]])

R2 = np.array([[9.96774172e-01, -6.17246664e-02, -5.12963500e-02],
              [6.18368289e-02, 9.98086091e-01, 6.00880232e-04],
              [5.11610843e-02, -3.77094552e-03, 9.98683295e-01]])

P1 = np.array([[378.63796211, 0, 368.74058151, 0],
               [0, 378.63796211, 255.47016335, 0],
               [0, 0, 1, 0]])

P2 = np.array([[3.78637962e+02, 0, 3.68740582e+02, 4.98233488e+03],
               [0, 3.78637962e+02, 2.55470163e+02, 0],
               [0, 0, 1, 0]])

map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, (w_l, h_l), cv2.CV_16SC2)
map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, (w_r, h_r), cv2.CV_16SC2)

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

    combined = np.hstack((undistorted_l, undistorted_r))
    y = combined.shape[0] // 2
    cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 2)

    # 实时预览
    cv2.imshow("Left Undistorted", undistorted_l)
    cv2.imshow("Right Undistorted", undistorted_r)
    cv2.imshow('Stereo View', combined)

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
