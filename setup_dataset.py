import os
import zipfile
from pathlib import Path
import shutil

def setup_drive_dataset():
    """
    Extract and organize DRIVE dataset for optic disc detection training.
    """
    
    # Paths
    base_dir = Path("Optic_Disc_Detection")
    downloads_dir = base_dir / "downloads"
    data_dir = base_dir / "data"
    
    # Look for DRIVE zip (various possible names)
    possible_zips = [
        "DRIVE.zip",
        "drive.zip",
        "training.zip",
        "test.zip"
    ]
    
    zip_path = None
    for zip_name in possible_zips:
        candidate = downloads_dir / zip_name
        if candidate.exists():
            zip_path = candidate
            break
    
    # Create directories
    os.makedirs(downloads_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    if not zip_path:
        print(f"❌ Error: DRIVE dataset ZIP not found in {downloads_dir}")
        print(f"   Looking for: {', '.join(possible_zips)}")
        print(f"\nPlease download the dataset first. See DATASET_DOWNLOAD.md")
        return False
    
    print(f"📦 Found: {zip_path.name}")
    print(f"📦 Extracting...")
    
    # Extract
    extract_dir = downloads_dir / "DRIVE_extracted"
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    print("✅ Extraction complete")
    
    # Organize files
    # DRIVE structure varies, try to find images
    print("\n📁 Organizing files...")
    
    training_dir = data_dir / "training"
    test_dir = data_dir / "test"
    
    os.makedirs(training_dir / "images", exist_ok=True)
    os.makedirs(training_dir / "masks", exist_ok=True)
    os.makedirs(test_dir / "images", exist_ok=True)
    os.makedirs(test_dir / "masks", exist_ok=True)
    
    # Find and copy files (adapt based on actual DRIVE structure)
    # This is a template - actual paths may vary
    train_count = 0
    test_count = 0
    
    # Search for image files
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            src = Path(root) / file
            
            # Training images
            if 'training' in root.lower() and file.endswith(('.tif', '.png', '.jpg')):
                if 'manual' in file.lower() or 'mask' in file.lower():
                    dest = training_dir / "masks" / file
                else:
                    dest = training_dir / "images" / file
                shutil.copy2(src, dest)
                train_count += 1
            
            # Test images
            elif 'test' in root.lower() and file.endswith(('.tif', '.png', '.jpg')):
                if 'manual' in file.lower() or 'mask' in file.lower():
                    dest = test_dir / "masks" / file
                else:
                    dest = test_dir / "images" / file
                shutil.copy2(src, dest)
                test_count += 1
    
    print(f"✅ Organized {train_count} training files")
    print(f"✅ Organized {test_count} test files")
    
    print("\n" + "="*50)
    print("✅ Dataset setup complete!")
    print("="*50)
    print(f"\n📁 Data structure:")
    print(f"   Training: {training_dir}")
    print(f"   Test:     {test_dir}")
    print(f"\n▶️  Next step: Run train.py to start training")
    
    return True

if __name__ == "__main__":
    success = setup_drive_dataset()
    if not success:
        print("\n⚠️  Setup failed. Please check the error messages above.")
        exit(1)
