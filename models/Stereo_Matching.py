import cv2
from ..utils.Visualization import *

class DepthMeasurementSystem:
    def __init__(self):
        self.base_left = None
        self.base_right = None
        self.last_valid = (None, None)
        self.cap_left = cv2.VideoCapture(0)
        self.cap_right = cv2.VideoCapture(1)
    
    def get_baseline(self):
        return self.last_valid if self.left is None else (self.left, self.right)
    
    def check_unpressed(self, frame, model):
        tensor = numpy_to_tensor(frame)  # 转换为模型输入格式
        
        with torch.no_grad():
            recon, threshold = self.model(tensor)
            error = torch.mean((tensor - recon)**2).item()
        
        return error < threshold.item()
    
    def process_frame(self):
        if not self.cap_left.isOpened() or not self.cap_right.isOpened():
            if not self.cap_left.isOpened():
                print("no left")
            if not self.cap_right.isOpened():
                print("no right")

                