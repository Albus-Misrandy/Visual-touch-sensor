"""
Author: Albus.Misrandy
Purpose: Stereo Camera Calibration using saved image pairs from Left_photos/ and Right_photos/
"""

import cv2
import numpy as np
import glob
import os

# 棋盘格设置（你之前是 11x8 内部角点）
chessboard_size = (11, 8)
square_size = 1.5  # 单位：mm（根据你实际棋盘格格子大小设置）

# 文件夹路径
left_folder = 'Left_photos'
right_folder = 'Right_photos'

# 获取图像路径列表并排序
left_images = sorted(glob.glob(os.path.join(left_folder, '*.jpg')))
right_images = sorted(glob.glob(os.path.join(right_folder, '*.jpg')))

assert len(left_images) == len(right_images), "左右图片数量不一致，请检查！"

# 准备棋盘格世界坐标（Z=0）
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0],
                       0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# 存储所有图像的角点
objpoints = []        # 世界坐标（3D）
imgpoints_left = []   # 左相机图像角点（2D）
imgpoints_right = []  # 右相机图像角点（2D）

image_size = None

for left_path, right_path in zip(left_images, right_images):
    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)

    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    if image_size is None:
        image_size = gray_left.shape[::-1]  # 获取图像大小

    # 查找角点
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

    if ret_left and ret_right:
        # 亚像素精细化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)
    else:
        print(f"跳过未检测到角点的图像对: {left_path}, {right_path}")

print("检测完成，开始标定...")

# 左右相机分别单目标定
ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, image_size, None, None)
ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, image_size, None, None)

# 双目标定
flags = cv2.CALIB_FIX_INTRINSIC  # 固定内参，只求外参
criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    mtx_l,
    dist_l,
    mtx_r,
    dist_r,
    image_size,
    criteria=criteria_stereo,
    flags=flags
)

print("\n标定完成！结果如下：")
print("左相机内参矩阵:\n", mtx_l)
print("右相机内参矩阵:\n", mtx_r)
print("左相机畸变系数:\n", dist_l)
print("右相机畸变系数:\n", dist_r)
print("旋转矩阵 R:\n", R)
print("平移向量 T:\n", T)

baseline_meters = np.linalg.norm(T)
print(f"基线距离: {baseline_meters:.3f} 米")

# 可选：立体校正
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtx_l, dist_l,
    mtx_r, dist_r,
    image_size,
    R, T
)

print("\nR1:", R1)
print("\nR2:", R2)
print("\nP1:", P1)
print("\nP2:", P2)
print("\n立体矫正矩阵 Q:\n", Q)
