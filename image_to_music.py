# from transformers import pipeline
# import scipy

# synthesiser = pipeline("text-to-audio", "facebook/musicgen-medium")

# music = synthesiser("The woman is scared of something", forward_params={"do_sample": True})

# scipy.io.wavfile.write("./musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
import os
from pathlib import Path
import torch
from PIL import Image
import scipy
import argparse
from tqdm import tqdm
from transformers import pipeline, Blip2Processor, Blip2ForConditionalGeneration
from typing import Optional, Callable

# Available models
AVAILABLE_CAPTION_MODELS = {
    "emo": "/home/ec2-user/project/emo-captioning-model",  # finetuned BLIP-2 model
    "blip2": "Salesforce/blip2-opt-2.7b",  # BLIP-2 
}

AVAILABLE_AUDIO_MODELS = {
    "musicgen-small": "facebook/musicgen-small",
    "musicgen-medium": "facebook/musicgen-medium",
    "musicgen-large": "facebook/musicgen-large",
    "audioldm2": "cvssp/audioldm2-music",
}

DEFAULT_IMAGES_DIR = Path("data/selected_images")
DEFAULT_OUT_DIR = Path("generated_music")
DEFAULT_CAPTION_MODEL = "emo"
DEFAULT_AUDIO_MODEL = "musicgen-small"

PIPELINE_KWARGS = {"device_map": "auto"}
FORWARD_PARAMS = {"do_sample": True}

def get_image_caption(processor, model, image_path):
    """Generate caption for a single image using the specified model"""
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    generated_ids = model.generate(**inputs, max_new_tokens=30)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return caption

def parse_args():
    parser = argparse.ArgumentParser(description="Generate music from images using AI models")
    parser.add_argument(
        "--image", 
        type=str,
        help="Path to a specific image to process. If not provided, processes all images in data/images"
    )
    parser.add_argument(
        "--caption-model",
        type=str,
        choices=list(AVAILABLE_CAPTION_MODELS.keys()),
        default=DEFAULT_CAPTION_MODEL,
        help=f"Captioning model to use. Available: {', '.join(AVAILABLE_CAPTION_MODELS.keys())}"
    )
    parser.add_argument(
        "--audio-model",
        type=str,
        choices=list(AVAILABLE_AUDIO_MODELS.keys()),
        default=DEFAULT_AUDIO_MODEL,
        help=f"Audio generation model to use. Available: {', '.join(AVAILABLE_AUDIO_MODELS.keys())}"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Base directory to save generated music files"
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------------
# helper to load an audio generator ------------------------------------------------
def load_audio_generator(model_key: str, device: str = "cuda") -> "Callable":
    """
    Return a callable that produces a dict with keys 'sampling_rate' and 'audio'
    from a text prompt.

    For MusicGen / Riffusion / Stable-Audio we keep the Transformers pipeline.
    For AudioLDM-2 we fall back to the Diffusers pipeline.
    """
    if model_key == "audioldm2":
        from diffusers import AudioLDM2Pipeline
        pipe = AudioLDM2Pipeline.from_pretrained(
            AVAILABLE_AUDIO_MODELS[model_key],
            torch_dtype=torch.float16,
        ).to(device)

        def run(prompt: str, forward_params: Optional[dict] = None):
            forward_params = forward_params or {}
            # map the params we used before to AudioLDM2 equivalents
            num_steps = forward_params.get("num_inference_steps", 200)
            length_s  = forward_params.get("audio_length_in_s", 10.0)
            audio = pipe(prompt,
                         num_inference_steps=num_steps,
                         audio_length_in_s=length_s).audios[0]
            return {"sampling_rate": 16000, "audio": audio}

        return run
    else:
        # legacy path – standard Transformers pipeline
        return pipeline("text-to-audio",
                        AVAILABLE_AUDIO_MODELS[model_key],
                        **PIPELINE_KWARGS)
# ---------------------------------------------------------------------------------

def main():
    args = parse_args()
    
    model_output_dir = Path(args.output_dir) / f"{args.caption_model}_{args.audio_model}"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading captioning model ({args.caption_model})...")
    processor = Blip2Processor.from_pretrained(AVAILABLE_CAPTION_MODELS[args.caption_model])
    caption_model = Blip2ForConditionalGeneration.from_pretrained(
        AVAILABLE_CAPTION_MODELS[args.caption_model],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    print(f"Loading audio generation model ({args.audio_model})...")
    synthesiser = load_audio_generator(args.audio_model,
                                       device="cuda" if torch.cuda.is_available() else "cpu")

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {args.image}")
        image_files = [image_path]
    else:
        image_files = list(DEFAULT_IMAGES_DIR.glob("*.jpg"))
        if not image_files:
            raise FileNotFoundError(f"No JPG images found in {DEFAULT_IMAGES_DIR}")

    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            caption = get_image_caption(processor, caption_model, img_path)
            print(f"\nImage: {img_path.name}")
            print(f"Caption: {caption}")

            prefix = img_path.stem
            out_wav = model_output_dir / f"{prefix}.wav"
            
            music = synthesiser(caption, forward_params=FORWARD_PARAMS)
            scipy.io.wavfile.write(
                out_wav,
                rate=music["sampling_rate"],
                data=music["audio"]
            )
            
        except Exception as e:
            print(f"[WARN] Failed on '{img_path.name}': {e}")

    print(f"\nAll done! WAV files saved to {model_output_dir.resolve()}")

if __name__ == "__main__":
    main()