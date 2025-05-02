import os
import random
import csv
from pathlib import Path
from typing import List

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Blip2ForConditionalGeneration

# ---------- Configuration ----------
MODEL_PATH   = "./emo-captioning-model"
IMAGE_DIR    = Path("data/images")
OUTPUT_CSV   = "captions.csv"
SAMPLE_SIZE  = 20           
IMAGE_EXTS   = {".jpg", ".jpeg", ".png"}
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------------


def load_model_and_processor(model_path: str):
    """Load BLIP-2 model and processor from a local folder."""
    processor = AutoProcessor.from_pretrained(model_path)
    model = Blip2ForConditionalGeneration.from_pretrained(model_path).to(DEVICE)
    model.eval()
    return model, processor


def pick_images(directory: Path, sample_size: int) -> List[Path]:
    """Return *sample_size* image paths from *directory* (random order)."""
    all_images = [
        p for p in directory.rglob("*")   # 支持子目录
        if p.suffix.lower() in IMAGE_EXTS
    ]
    if len(all_images) == 0:
        raise FileNotFoundError(f"No images with extensions {IMAGE_EXTS} found in {directory}")
    # 随机抽样；若想按文件名顺序取前 N 张，去掉 random.choice
    return random.sample(all_images, min(sample_size, len(all_images)))


@torch.inference_mode()
def generate_caption(image_path: Path, model, processor) -> str:
    """Generate a caption for a single image file."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        max_length=50,
        num_beams=5,
        length_penalty=1.0,
        temperature=1.0,
    )
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return caption


def main():
    # 1) Load model & processor
    print("Loading model …")
    model, processor = load_model_and_processor(MODEL_PATH)

    # 2) Choose images
    images = pick_images(IMAGE_DIR, SAMPLE_SIZE)
    print(f"Found {len(images)} images for captioning.")

    # 3) Generate captions
    captions = []
    for img_path in tqdm(images, desc="Generating captions"):
        caption = generate_caption(img_path, model, processor)
        captions.append((img_path.name, caption))

    # 4) Save to CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "caption"])
        writer.writerows(captions)

    print(f"\n Captions saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()