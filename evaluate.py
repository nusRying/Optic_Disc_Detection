import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np

from src.dataset import RetinalDataset
from src.model import UNet

def calculate_metrics(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2. * intersection) / (union + 1e-8)
    iou = intersection / (union - intersection + 1e-8)
    
    return dice.item(), iou.item()

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = RetinalDataset(args.images_dir, args.masks_dir, img_size=(args.img_size, args.img_size))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    total_dice = 0
    total_iou = 0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            dice, iou = calculate_metrics(outputs, masks)
            
            total_dice += dice
            total_iou += iou
            
    print(f"Average Dice Score: {total_dice/len(loader):.4f}")
    print(f"Average IoU: {total_iou/len(loader):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True, help='Path to test images')
    parser.add_argument('--masks_dir', type=str, required=True, help='Path to test masks')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--img_size', type=int, default=256)
    
    args = parser.parse_args()
    evaluate(args)
