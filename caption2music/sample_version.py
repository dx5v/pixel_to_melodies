import os
from pathlib import Path
import torch
from PIL import Image
import scipy
from tqdm import tqdm
from transformers import pipeline, Blip2Processor, Blip2ForConditionalGeneration

# ---------- config ----------
IMAGES_DIR = Path("data/images")
OUT_DIR = Path("generated_music")
BLIP_MODEL = "Salesforce/blip2-opt-2.7b"
MUSIC_MODEL = "facebook/musicgen-small"  # Changed to small model
PIPELINE_KWARGS = {"device_map": "auto"}
FORWARD_PARAMS = {"do_sample": True}
# ---------------------------

def get_image_caption(processor, model, image_path):
    """Generate caption for a single image using BLIP-2"""
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    generated_ids = model.generate(**inputs, max_new_tokens=30)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return caption

def main():
    print("Loading BLIP-2 model...")
    processor = Blip2Processor.from_pretrained(BLIP_MODEL)
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP_MODEL, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    print("Loading MusicGen model...")
    synthesiser = pipeline("text-to-audio", MUSIC_MODEL, **PIPELINE_KWARGS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = list(IMAGES_DIR.glob("*.jpg"))
    if not image_files:
        raise FileNotFoundError(f"No JPG images found in {IMAGES_DIR}")

    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            caption = get_image_caption(processor, blip_model, img_path)
            print(f"\nImage: {img_path.name}")
            print(f"Caption: {caption}")

            prefix = img_path.stem
            out_wav = OUT_DIR / f"{prefix}.wav"
            
            music = synthesiser(caption, forward_params=FORWARD_PARAMS)
            scipy.io.wavfile.write(
                out_wav,
                rate=music["sampling_rate"],
                data=music["audio"]
            )
            
        except Exception as e:
            print(f"[WARN] Failed on '{img_path.name}': {e}")

    print(f"\nAll done! WAV files saved to {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main() 