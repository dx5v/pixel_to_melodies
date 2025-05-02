import os
import random
import shutil
from pathlib import Path

# Configuration
SOURCE_DIR = Path("data/images")
TARGET_DIR = Path("data/selected_images")
NUM_IMAGES = 50

def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    image_files = list(SOURCE_DIR.glob("*.jpg"))
    
    if len(image_files) < NUM_IMAGES:
        print(f"Warning: Only {len(image_files)} images available, less than requested {NUM_IMAGES}")
        selected_images = image_files
    else:
        selected_images = random.sample(image_files, NUM_IMAGES)
    
    for img_path in selected_images:
        target_path = TARGET_DIR / img_path.name
        shutil.copy2(img_path, target_path)
        print(f"Copied: {img_path.name}")
    
    print(f"\nDone! {len(selected_images)} images copied to {TARGET_DIR.resolve()}")

if __name__ == "__main__":
    main() 