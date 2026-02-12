import torch
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.model import UNet

def preprocess(image_path, img_size):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # Resize
    image = cv2.resize(image, (img_size, img_size))
    
    # Normalize
    image = image / 255.0
    
    # Tensor
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.from_numpy(image).unsqueeze(0)
    
    return image

def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    image_tensor = preprocess(args.image_path, args.img_size).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        
    mask = output.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    
    # Resize mask back to original image size for overlay (optional, but good for display)
    original_image = cv2.imread(args.image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original_image)
    ax[0].set_title("Input Image" + (" (CLAHE applied)" if False else "")) # Note: showing original here
    ax[0].axis('off')
    
    ax[1].imshow(mask_resized, cmap='gray')
    ax[1].set_title("Predicted Optic Disc")
    ax[1].axis('off')
    
    if args.save:
        plt.savefig('output_detection.png')
        print("Result saved to 'output_detection.png'")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to test image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--save', action='store_true', help='Save output instead of showing')
    
    args = parser.parse_args()
    inference(args)
