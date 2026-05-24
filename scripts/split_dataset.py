from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

SOURCE_DIR = Path("../data/raw")
OUTPUT_DIR = Path("../data")
SEED = 1

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-8

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split_name in ["train", "val", "test"]:
    (OUTPUT_DIR / split_name).mkdir(parents=True, exist_ok=True)

for class_dir in SOURCE_DIR.iterdir():
    if not class_dir.is_dir():
        continue

    image_paths = sorted(
        [p for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]
    )

    if len(image_paths) < 3:
        print(f"Skipping {class_dir.name}: not enough images to split.")
        continue

    train_files, temp_files = train_test_split(
        image_paths,
        train_size=TRAIN_RATIO,
        random_state=SEED,
        shuffle=True
    )

    val_relative_ratio = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    val_files, test_files = train_test_split(
        temp_files,
        train_size=val_relative_ratio,
        random_state=SEED,
        shuffle=True
    )

    split_map = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    for split_name, files in split_map.items():
        target_class_dir = OUTPUT_DIR / split_name / class_dir.name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        for file_path in files:
            shutil.copy2(file_path, target_class_dir / file_path.name)

print("Stratified-style folder split completed.")