import os
import shutil
from pathlib import Path

test_path = Path("data/test")
organized_path = Path("data/test_cleaned")

organized_path.mkdir(exist_ok=True)

for file in test_path.iterdir():
    if file.is_file() and file.suffix.lower() in [".jpg", ".png", ".jpeg"]:
        label = file.name.split("_")[0]
        label_folder = organized_path / label
        label_folder.mkdir(exist_ok=True)
        shutil.copy(file, label_folder / file.name)

print("Test images reorganized into class folders.")