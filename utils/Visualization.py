import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def numpy_to_tensor(img):
    img_numpy = np.array(img)  # 从img转换为NumPy数组
    if img_numpy.ndim == 3:
        img_numpy = img_numpy.transpose((2, 0, 1))  # （可选）通道变换 (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(img_numpy).float()  # 将图像从numpy数组转换为PyTorch张量
        tensor = tensor.to(torch.float32) / 255.0  # （可选）在0-1范围内标准化图像
        return tensor
    if img_numpy.ndim == 2:
        img_numpy = np.expand_dims(img_numpy, axis=0)  # （可选）通道变换 (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(img_numpy).float()  # 将图像从numpy数组转换为PyTorch张量
        tensor = tensor.to(torch.float32) / 255.0  # （可选）在0-1范围内标准化图像
        return tensor


# PyTorch张量转换为numpy
def tensor_to_numpy(tensor):
    # 将张量转换为numpy数组, 如果张量在GPU上，请先使用.cpu()将其移动到CPU。
    image_array = tensor.detach().cpu().clone().numpy()
    # image_array = tensor.detach().cpu().numpy()
    # （可选）如果图像是在0-1之间标准化的，则需要先反标准化（乘以255）
    image_array = (image_array * 255).astype(np.uint8)
    # image_array = np.clip(image_array, 0, 255).astype(np.uint8)  # 区间 0-255 截断
    # （可选）如果张量进行了通道变换则要变换回来 (C, H, W) -> (H, W, C)
    if image_array.shape[0] == 3:
        image_array = image_array.transpose((1, 2, 0))
    elif image_array.shape[0] == 1:
        image_array = image_array.squeeze(0)
    # print(image_array.shape)
    return image_array


def plot_muti_curve(x_data, y_data, title, xlabel, ylabel, colors=None, labels=None):
    """
    在同一张图上绘制多条曲线
    参数:
        x_data (list of lists): 多个x数据集，例如 [[x1], [x2], ...]
        y_data (list of lists): 多个y数据集，例如 [[y1], [y2], ...]
        title (str): 图表标题
        xlabel (str): X轴标签
        ylabel (str): Y轴标签
        colors (list of str, 可选): 每条曲线的颜色，例如 ["red", "blue"]
        labels (list of str, 可选): 每条曲线的图例标签
    """
    # 处理单条曲线的情况（保持兼容性）
    if not isinstance(x_data[0], (list, tuple, np.ndarray)):
        x_data = [x_data]
    if not isinstance(y_data[0], (list, tuple, np.ndarray)):
        y_data = [y_data]

    # 检查数据长度是否一致
    if len(x_data) != len(y_data):
        raise ValueError("x_data和y_data的曲线数量不一致")

    # 设置默认颜色和标签
    n_curves = len(x_data)
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if colors is None:
        colors = default_colors[:n_curves]
    else:
        colors += default_colors[len(colors):n_curves]  # 补充不足的颜色

    if labels is None:
        labels = [f"Curve {i + 1}" for i in range(n_curves)]

    # 绘制所有曲线
    for i in range(n_curves):
        plt.plot(x_data[i], y_data[i], color=colors[i], label=labels[i])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)  # 可选：添加网格线
    plt.show()

def crop_pictures(img, mask):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # # 转灰度图
    # gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    # 二值化：把白色区域提取出来
    _, binary = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("未找到任何轮廓")
        return None
    
    # 找面积最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 获取最小外接矩形
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # 裁剪图像
    cropped = img[y:y+h, x:x+w]
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    cropped = Image.fromarray(cropped)
    cropped = cropped.resize((640, 480))
    cropped.show()
        