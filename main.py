import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from Dataset.Dataset import SegmentationDataset
from models.Light_UNet import LightweightUNet

parser = argparse.ArgumentParser(description="Vision Touch Sensor.")

parser.add_argument("--origin_path", type=str, default="Dataset/Silicone_segment/Original",
                    help="Original photos' path.")
parser.add_argument("--mask_path", type=str, default="Dataset/Silicone_segment/labels", help="Mask photos' path.")
parser.add_argument("--batch_size", type=int, default=8, help="Value of batch size")
parser.add_argument("--Iterations", type=int, default=5, help="Training iterations.")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Value of Learning rate.")
parser.add_argument("--model_path", type=str, default="models/Best_BiSeNet.pth", help="Model path.")

args = parser.parse_args()


def Segment_train(device, epochs, train_loader, model, optimizer, criterion):
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            # print(images.shape)

            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算统计量
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f}')

    torch.save(model.state_dict(), args.model_path)


def predict(device, model, model_path, image_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.sigmoid(output)
        mask = (pred > 0.5).float()

        # 转换为可保存的numpy数组
        mask_np = mask[0][0].cpu().numpy()  # 去除batch和channel维度
        mask_np = (mask_np * 255).astype(np.uint8)  # 转为0-255的uint8格式

        # 创建并保存mask图像
        mask_image = Image.fromarray(mask_np)
        mask_image.show("Mask")


if __name__ == '__main__':
    # Define the parameters of the model training.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    train_dataset = SegmentationDataset(args.origin_path, args.mask_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    Segment_train(device, args.Iterations, train_loader, model, optimizer, criterion)
    # predict(device, model, args.model_path, "Image_Sampling/captured_images/58.jpg")
