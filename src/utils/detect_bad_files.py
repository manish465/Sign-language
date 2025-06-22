from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def scan_images(folder):
    bad = []
    for path in Path(folder).rglob("*.*"):
        print(path)
        if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]: continue
        try:
            with open(path, "rb") as f:
                img = Image.open(f)
                img.verify()
        except Exception as e:
            print(f"❌ {path} — {e}")
            bad.append(path)
    return bad

# Run this
bad_files = scan_images("data/train")
print(f"\n{len(bad_files)} bad images found.")
