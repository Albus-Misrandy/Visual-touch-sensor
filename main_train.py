import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from Dataset.Dataset import UnpressImageDataset
# from models.Light_UNet import LightweightUNet
from models.LiteTactileNet import LiteTactileNet
from utils.Visualization import *

parser = argparse.ArgumentParser(description="Vision Touch Sensor.")

parser.add_argument("--data_path", type=str, default="Image_Sampling/captured_images", help="Unpressed image data.")
parser.add_argument("--batch_size", type=int, default=16, help="Value of batch size")
parser.add_argument("--Iterations", type=int, default=100, help="Training iterations.")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Value of Learning rate.")
parser.add_argument("--model_path", type=str, default="models/FastTactileNet.pth", help="Model path.")

args = parser.parse_args()


def train_LiteTactileNet(device, epochs, train_loader, model, optimizer, recon_loss, threshold_loss):
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for images in train_loader:
            images = images.to(device)
            with torch.cuda.amp.autocast():
                recon, threshold = model(images)

                loss_recon = recon_loss(recon, images)
                
                # 动态阈值目标：μ+2σ原则
                errors = torch.mean((recon-images)**2, dim=[1,2,3])
                target = errors.mean() + 2 * errors.std()
                loss_threshold = threshold_loss(threshold.squeeze(), target)
                
                total_loss = loss_recon + 0.5*loss_threshold

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += total_loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f}')
        torch.cuda.empty_cache()

    val = evaluate_reconstruction(model, train_loader, device)
    print(f"MSE:{val['mse_mean']:.4f}±{val['mse_std']:.4f}")
    print(f"SSIM: {val['ssim_mean']:.4f}±{val['ssim_std']:.4f}")
    torch.save(model.state_dict(), args.model_path)


def evaluate_reconstruction(model, dataloader, device):
    model.eval()
    mse_losses = []
    ssim_scores = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch.to(device)
            reconstructions, _ = model(inputs)

            # 计算MSE
            mse = torch.mean((inputs - reconstructions) ** 2, dim=[1, 2, 3])
            mse_losses.extend(mse.cpu().numpy())

            # 计算SSIM
            inputs_np = inputs.cpu().numpy().transpose(0, 2, 3, 1)
            recon_np = reconstructions.cpu().numpy().transpose(0, 2, 3, 1)

            for i in range(inputs_np.shape[0]):
                # 动态计算窗口大小
                H, W, _ = inputs_np[i].shape
                win_size = min(7, H, W)
                win_size = win_size if win_size % 2 else win_size - 1  # 确保奇数
                win_size = max(win_size, 3)  # 最小3x3

                ssim_val = ssim(
                    inputs_np[i],
                    recon_np[i],
                    win_size=win_size,
                    channel_axis=-1,  # 替换multichannel参数
                    data_range=1.0
                )
                ssim_scores.append(ssim_val)

    return {
        'mse_mean': np.mean(mse_losses),
        'mse_std': np.std(mse_losses),
        'ssim_mean': np.mean(ssim_scores),
        'ssim_std': np.std(ssim_scores)
    }

def Segment_train(device, epochs, train_loader, model, optimizer, criterion):
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            # print(images.shape)
            # print(masks.shape)

            outputs = model(images)
            # print(outputs.shape)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算统计量
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f}')

        torch.cuda.empty_cache()

    torch.save(model.state_dict(), args.model_path)


def predict(device, model, model_path, image_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 开启cudnn加速
    # print(torch.backends.cudnn.benchmark)
    torch.backends.cudnn.benchmark = True

    image = Image.open(image_path).convert('RGB')
    image_tensor = numpy_to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        output = output.squeeze(0)
        pred = torch.sigmoid(output)
        mask = (pred > 0.1).float()

        mask_np = tensor_to_numpy(mask)

        # # 创建并保存mask图像
        mask_image = Image.fromarray(mask_np)
        mask_image.show("Mask")
        # mask_image.save("test.png")
        return mask_np
    

if __name__ == '__main__':
    torch.cuda.empty_cache()
    # Define the parameters of the model training.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # 启用cudnn自动优化
    model = LiteTactileNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    # 损失函数
    recon_loss = nn.L1Loss()  # 对重建任务更鲁棒
    threshold_loss = nn.MSELoss()

    train_dataset = UnpressImageDataset(args.data_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    train_LiteTactileNet(device, args.Iterations, train_loader, model, optimizer, recon_loss, threshold_loss)

#     # Segment_train(device, args.Iterations, train_loader, model, optimizer, criterion)
#     img = predict(device, model, args.model_path, "Image_Sampling/captured_images/56.jpg")
#     ori = Image.open("Image_Sampling/captured_images/56.jpg").convert('RGB')
#     crop_pictures(ori, img)
