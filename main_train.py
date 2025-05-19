import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from Dataset.Dataset import SegmentationDataset
from models.Light_UNet import LightweightUNet
from utils.Visualization import *

parser = argparse.ArgumentParser(description="Vision Touch Sensor.")

parser.add_argument("--origin_path", type=str, default="Dataset/Silicone_segment/Original",
                    help="Original photos' path.")
parser.add_argument("--mask_path", type=str, default="Dataset/Silicone_segment/labels", help="Mask photos' path.")
parser.add_argument("--batch_size", type=int, default=8, help="Value of batch size")
parser.add_argument("--Iterations", type=int, default=30, help="Training iterations.")
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
    model = LightweightUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    train_dataset = SegmentationDataset(args.origin_path, args.mask_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Segment_train(device, args.Iterations, train_loader, model, optimizer, criterion)
    img = predict(device, model, args.model_path, "Image_Sampling/captured_images/56.jpg")
    ori = Image.open("Image_Sampling/captured_images/56.jpg").convert('RGB')
    crop_pictures(ori, img)
