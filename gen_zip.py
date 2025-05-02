import os
import zipfile
from pathlib import Path
import pandas as pd

CSV_FILE = "captions.csv"
IMAGE_DIR = Path("selected_images")
AUDIO_DIR = Path("generated_music/blip2_musicgen-small")
ZIP_PATH = Path("blip2_pairs.zip")

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    # add CSV if present
    if Path(CSV_FILE).exists():
        zipf.write(CSV_FILE, arcname=Path(CSV_FILE).name)
        df = pd.read_csv(CSV_FILE)
        rows = df.to_dict("records")
    else:
        rows = []
    
    for row in rows:
        img_name = row["image"]
        image_path = IMAGE_DIR / img_name
        audio_path = AUDIO_DIR / f"{Path(img_name).stem}.wav"
        
        if image_path.exists():
            zipf.write(image_path, arcname=f"images/{img_name}")
        if audio_path.exists():
            zipf.write(audio_path, arcname=f"audio/{audio_path.name}")
