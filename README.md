# Optic Disc Detection using U-Net

This project implements a Deep Learning pipeline for detecting the Optic Disc in retinal images.

## Setup

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare Dataset:
   - Download DRIVE or STARE dataset.
   - Place images in `data/images` and binary masks in `data/masks`.

## Usage

### Training

Train the U-Net model:

```bash
python train.py --images_dir data/images --masks_dir data/masks --epochs 50 --batch_size 4
```

Checkpoints will be saved in `checkpoints/`.

### Evaluation

Evaluate the model using Dice Score and IoU:

```bash
python evaluate.py --images_dir data/images --masks_dir data/masks --model_path checkpoints/best_model.pth
```

### Inference

Run inference on a single image:

```bash
python inference.py --image_path path/to/image.jpg --model_path checkpoints/best_model.pth --save
```

This will save the result as `output_detection.png`.
