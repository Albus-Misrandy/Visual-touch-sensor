"""
Author: Albus.Misrandy
Updated: Dual Camera Capture with Chessboard Detection (Clone Frame Version)
"""
import cv2
import os

# 参数设置
chessboard_size = (11, 8)   # 棋盘格内角点数量
save_dir_left = 'Left_photos'
save_dir_right = 'Right_photos'

# 创建文件夹
os.makedirs(save_dir_left, exist_ok=True)
os.makedirs(save_dir_right, exist_ok=True)

# 打开两个摄像头
cap_left = cv2.VideoCapture(1)
cap_right = cv2.VideoCapture(0)

if not cap_left.isOpened():
    print("无法打开摄像头1")
    exit()

if not cap_right.isOpened():
    print("无法打开摄像头2")
    exit()

print("按 'c' 键保存当前帧（不包含角点标记），按 'ESC' 键退出")

image_count = 0

while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        print("无法获取两个摄像头的帧")
        break

    # --- 拷贝一份frame用于显示 ---
    display_left = frame_left.copy()
    display_right = frame_right.copy()

    # 转灰度图
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret_corners_left, corners_left = cv2.findChessboardCorners(
        gray_left, chessboard_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    ret_corners_right, corners_right = cv2.findChessboardCorners(
        gray_right, chessboard_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    # 如果检测到，就画在display副本上
    if ret_corners_left:
        cv2.drawChessboardCorners(display_left, chessboard_size, corners_left, ret_corners_left)
    if ret_corners_right:
        cv2.drawChessboardCorners(display_right, chessboard_size, corners_right, ret_corners_right)

    # 显示处理过的图像
    cv2.imshow("Left Camera", display_left)
    cv2.imshow("Right Camera", display_right)

    key = cv2.waitKey(1) & 0xFF

    # 按 'c' 键保存（保存的是原始帧）
    if key == ord('c'):
        image_count += 1
        left_img_name = os.path.join(save_dir_left, f"{image_count:04d}.jpg")
        right_img_name = os.path.join(save_dir_right, f"{image_count:04d}.jpg")
        cv2.imwrite(left_img_name, frame_left)  # 保存的是原始frame
        cv2.imwrite(right_img_name, frame_right)
        print(f"保存成功: 左图->{left_img_name}, 右图->{right_img_name}")

    # 按 'ESC' 键退出
    elif key == 27:
        print("退出程序")
        break

# 释放资源
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
