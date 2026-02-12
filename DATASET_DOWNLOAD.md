# DRIVE Dataset Setup

## Important Notice
The DRIVE (Digital Retinal Images for Vessel Extraction) dataset requires **registration** before download.

---

## Registration & Download

### Step 1: Register
1. **Visit**: https://drive.grand-challenge.org/
2. **Click** "Request Dataset Access"
3. **Fill form** with your details (academic/research purpose)
4. **Wait** for approval email (usually 24-48 hours)

### Step 2: Download
1. **Login** to Grand Challenge
2. **Download** the dataset ZIP file (~2MB)
3. **Save to**: `c:/Users/umair/Videos/CV Project 2/Optic_Disc_Detection/downloads/`

---

## After Download

Run the setup script:
```bash
cd "c:/Users/umair/Videos/CV Project 2"
conda activate lcs
python Optic_Disc_Detection/setup_dataset.py
```

This will:
1. Extract images and masks
2. Organize into train/test splits
3. Prepare for model training

---

## Dataset Structure (After Setup)

```
Optic_Disc_Detection/
├── downloads/
│   └── DRIVE.zip
├── data/
│   ├── training/
│   │   ├── images/     (20 fundus images)
│   │   └── masks/      (manual segmentations)
│   └── test/
│       ├── images/     (20 fundus images)
│       └── masks/
```

---

## Dataset Details
- **Total images**: 40 retinal fundus photographs
- **Resolution**: 565×584 pixels, 8-bit RGB
- **Split**: 20 training + 20 test
- **Annotations**: Manual vessel segmentation + optic disc masks
- **Source**: Diabetic retinopathy screening (Netherlands)

---

## Alternative: Download Now (No Registration)
If you need immediate access for testing, you can use the STARE dataset which is publicly available:

**STARE Project**: http://cecas.clemson.edu/~ahoover/stare/
- 400 images (80 with optic disc ground truth)
- Larger variety, more challenging cases
- Direct download without registration
