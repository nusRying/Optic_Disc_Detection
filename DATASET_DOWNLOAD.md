# DRIVE Dataset Setup

This project uses the DRIVE retinal dataset for optic disc detection.

Run commands from the repository root:

```bash
cd "c:/Users/umair/Videos/CV Project 2"
```

## Access and Download

1. Open https://drive.grand-challenge.org/
2. Request dataset access.
3. Wait for approval email.
4. Download the dataset zip archive.
5. Place the zip in `Optic_Disc_Detection/downloads/`.

The setup script looks for these names:

- `DRIVE.zip`
- `drive.zip`
- `training.zip`
- `test.zip`

## Run Setup

```bash
python Optic_Disc_Detection/setup_dataset.py
```

## Expected Output Structure

```
Optic_Disc_Detection/
|-- downloads/
|   `-- DRIVE.zip
`-- data/
    |-- training/
    |   |-- images/
    |   `-- masks/
    `-- test/
        |-- images/
        `-- masks/
```

## Dataset Notes

- 40 retinal fundus images total (20 train, 20 test)
- Image resolution is approximately 565x584
- Registration is required before download

## Next Steps

```bash
python Optic_Disc_Detection/train.py --images_dir Optic_Disc_Detection/data/training/images --masks_dir Optic_Disc_Detection/data/training/masks
python Optic_Disc_Detection/evaluate.py --images_dir Optic_Disc_Detection/data/test/images --masks_dir Optic_Disc_Detection/data/test/masks --model_path checkpoints/best_model.pth
```
