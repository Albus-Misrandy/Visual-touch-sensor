import torch
# from threading import Thread
# from queue import Queue
from utils.Visualization import *
from models.LiteTactileNet import LiteTactileNet


class DepthMeasurementSystem:
    def __init__(self, model_path):
        self.base_left = None
        self.base_right = None
        self.last_valid_baseline = (None, None)
        self.cap_left = cv2.VideoCapture(0)
        self.cap_right = cv2.VideoCapture(1)
        self.mtx_l = np.array([[377.62484394, 0, 338.22935123],
                               [0, 381.64291743, 248.49561413],
                               [0, 0, 1]])
        self.dist_l = np.array([-0.25887591, 0.00739472, 0.00184546, -0.00079176, 0.02526219])
        self.mtx_r = np.array([[372.31843353, 0, 343.82837719],
                               [0, 375.63300679, 253.46532185],
                               [0, 0, 1]])
        self.dist_r = np.array([-0.2614851, 0.06556446, -0.0011021, -0.00212242, -0.0439001])
        self.R1 = np.array([[0.99902054, -0.03584055, -0.02595043],
                            [0.03578368, 0.99935604, -0.00265274],
                            [0.02602879, 0.00172154, 0.99965971]])
        self.R2 = np.array([[9.96774172e-01, -6.17246664e-02, -5.12963500e-02],
                            [6.18368289e-02, 9.98086091e-01, 6.00880232e-04],
                            [5.11610843e-02, -3.77094552e-03, 9.98683295e-01]])
        self.P1 = np.array([[378.63796211, 0, 368.74058151, 0],
                            [0, 378.63796211, 255.47016335, 0],
                            [0, 0, 1, 0]])
        self.P2 = np.array([[3.78637962e+02, 0, 3.68740582e+02, 4.98233488e+03],
                            [0, 3.78637962e+02, 2.55470163e+02, 0],
                            [0, 0, 1, 0]])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LiteTactileNet().to(self.device)
        self.model_path = model_path

        self.window_size = 5
        # 左视差计算器
        self.stereo_left = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=5,
            P1=8 * 3 * 5 ** 2,
            P2=32 * 3 * 5 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10
        )

        # 右视差计算器
        self.stereo_right = cv2.ximgproc.createRightMatcher(self.stereo_left)
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.stereo_left)
        self.wls_filter.setLambda(8000)
        self.wls_filter.setSigmaColor(1.5)

        self.focal_length = 379.633880685
        self.baseline_distance = 0.013159

    def get_baseline(self):
        return self.last_valid_baseline if self.base_left is None else (self.base_left, self.base_right)

    def check_unpressed(self, frame):
        tensor = numpy_to_tensor(frame, self.device)
        tensor = tensor.unsqueeze(0)  # 转换为模型输入格式
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        with torch.no_grad():
            recon, threshold = self.model(tensor)
            error = torch.mean((tensor - recon) ** 2).item()

        return error < threshold.item()

    def process_frame(self):
        if not self.cap_left.isOpened() or not self.cap_right.isOpened():
            if not self.cap_left.isOpened():
                print("no left")
                return
            if not self.cap_right.isOpened():
                print("no right")
                return

        ret1, frame_left = self.cap_left.read()
        ret2, frame_right = self.cap_right.read()

        h_l, w_l = frame_left.shape[:2]
        h_r, w_r = frame_right.shape[:2]

        map1_l, map2_l = cv2.initUndistortRectifyMap(self.mtx_l, self.dist_l, self.R1, self.P1, (w_l, h_l),
                                                     cv2.CV_16SC2)
        map1_r, map2_r = cv2.initUndistortRectifyMap(self.mtx_r, self.dist_r, self.R2, self.P2, (w_r, h_r),
                                                     cv2.CV_16SC2)
        return map1_l, map2_l, map1_r, map2_r

    def depth_computing(self, undistorted_l, undistorted_r):
        self.base_left = cv2.resize(self.last_valid_baseline[0], (320, 240))
        self.base_right = cv2.resize(self.last_valid_baseline[1], (320, 240))
        curr_left = cv2.resize(undistorted_l, (320, 240))
        curr_right = cv2.resize(undistorted_r, (320, 240))

        # 计算视差
        disparity_left = self.stereo_left.compute(curr_left, curr_right).astype(np.float32) / 16.0
        disparity_right = self.stereo_right.compute(curr_right, curr_left).astype(np.float32) / 16.0
        filtered_disp = self.wls_filter.filter(disparity_left, curr_left, None, disparity_right)

        # 计算深度
        depth = (self.focal_length * self.baseline_distance) / (filtered_disp + 1e-6)
        depth = cv2.resize(depth, (640, 480))  # 恢复原始分辨率
        return depth
