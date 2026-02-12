# Implementation Plan: Optic Disc Detection using U-Net

## Goal Description
Develop a Deep Learning pipeline to detect and segment the Optic Disc in retinal images (e.g., from DRIVE or STARE datasets) using a U-Net architecture.

## Proposed Changes

### Directory Structure
We will create a specific folder `Optic_Disc_Detection` within the project root.

```
Optic_Disc_Detection/
├── data/                   # Placeholders for DRIVE/STARE datasets
│   ├── images/
│   └── masks/
├── src/
│   ├── __init__.py
│   ├── dataset.py          # PyTorch Dataset class
│   ├── model.py            # U-Net Architecture
│   └── utils.py            # CLAHE, Metrics (Dice/IoU)
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── inference.py            # Inference on single images
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

### Components

#### 1. Data Pipeline (`src/dataset.py`)
- **Class**: `RetinalDataset(Dataset)`
- **Functionality**:
    - Load images and corresponding binary masks.
    - Apply **preprocessing**:
        - **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to enhance retinal features.
        - Resize to fixed input size (e.g., 256x256 or 512x512).
        - Normalization (0-1).
    - Data Augmentation (optional but recommended: rotation, flip).

#### 2. Model Architecture (`src/model.py`)
- **Class**: `UNet(nn.Module)`
- **Structure**:
    - **Encoder**: Convolutional blocks + MaxPool (Contracting path).
    - **Bottleneck**: Deepest features.
    - **Decoder**: UpConv/TransposedConv + Concatenation with Encoder features (Expanding path).
    - **Output**: 1x1 Conv + Sigmoid activation for binary segmentation.

#### 3. Training (`train.py`)
- **Loss Function**: `DiceLoss` (better for segmentation than BCE) + `BCELoss` (optional hybrid).
- **Optimizer**: Adam.
- **Loop**: Train over epochs, save best model based on validation loss.

#### 4. Evaluation & Inference (`evaluate.py`, `inference.py`)
- Calculate **Dice Coefficient** and **IoU (Intersection over Union)**.
- Visualize Input Image, Ground Truth, and Predicted Mask side-by-side.

## Verification Plan
### Automated Verification
- We will try to run the training loop with dummy random data if real data is not present, to ensure the pipeline (forward/backward pass) works.
- Check input/output tensor shapes.

### Manual Verification
- User will be instructed to place DRIVE/STARE dataset images in `data/`.
- Run `python inference.py --img <path>` to see segmentation results.
