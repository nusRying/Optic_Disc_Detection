import os
import shutil
import zipfile
from pathlib import Path


def setup_drive_dataset() -> bool:
    """Extract and organize DRIVE dataset into train/test image-mask structure."""
    script_dir = Path(__file__).resolve().parent
    downloads_dir = script_dir / "downloads"
    data_dir = script_dir / "data"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    possible_zips = ["DRIVE.zip", "drive.zip", "training.zip", "test.zip"]
    zip_path = None
    for name in possible_zips:
        candidate = downloads_dir / name
        if candidate.exists():
            zip_path = candidate
            break

    if zip_path is None:
        print(f"Error: DRIVE archive not found in {downloads_dir}")
        print(f"Expected one of: {', '.join(possible_zips)}")
        print("See Optic_Disc_Detection/DATASET_DOWNLOAD.md for details.")
        return False

    extract_dir = downloads_dir / "DRIVE_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found archive: {zip_path.name}")
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    training_dir = data_dir / "training"
    test_dir = data_dir / "test"
    for split_dir in [training_dir, test_dir]:
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "masks").mkdir(parents=True, exist_ok=True)

    train_count = 0
    test_count = 0
    for root, _, files in os.walk(extract_dir):
        root_lower = root.lower()
        for file in files:
            if not file.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                continue
            src = Path(root) / file
            is_mask = ("manual" in file.lower()) or ("mask" in file.lower())

            if "training" in root_lower:
                dest = training_dir / ("masks" if is_mask else "images") / file
                shutil.copy2(src, dest)
                train_count += 1
            elif "test" in root_lower:
                dest = test_dir / ("masks" if is_mask else "images") / file
                shutil.copy2(src, dest)
                test_count += 1

    print("Dataset setup complete.")
    print(f"Training files organized: {train_count}")
    print(f"Test files organized: {test_count}")
    print(f"Training dir: {training_dir}")
    print(f"Test dir: {test_dir}")
    return True


if __name__ == "__main__":
    ok = setup_drive_dataset()
    if not ok:
        raise SystemExit(1)
