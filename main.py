import cv2
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


if __name__ == "__main__":
    cap_left = cv2.VideoCapture(0)
    cap_right = cv2.VideoCapture(1)

    

    if not cap_left.isOpened() or not cap_right.isOpened():
        if not cap_left.isOpened():
            print("no left")
        if not cap_right.isOpened():
            print("right")

    ret1, frame_left = cap_left.read()
    ret2, frame_right = cap_right.read()

    h_l, w_l = frame_left.shape[:2]
    h_r, w_r = frame_right.shape[:2]

    map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, (w_l, h_l), cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, (w_r, h_r), cv2.CV_16SC2)

    while True:
        ret1, frame_l = cap_left.read()
        ret2, frame_r = cap_right.read()

        if not ret1 or not ret2:
            print("图像捕获失败")
            break

        undistorted_l = cv2.remap(frame_l, map1_l, map2_l, interpolation=cv2.INTER_LINEAR)
        undistorted_r = cv2.remap(frame_r, map1_r, map2_r, interpolation=cv2.INTER_LINEAR)
