import os, sys, csv
from pathlib import Path
import librosa, numpy as np, simplejpeg, torch, torchvision as tv
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.abspath(f'{os.getcwd()}/..'))
from model import AudioCLIP
from utils.transforms import ToTensor1D
# ----------------------------------

# ---------------- config ----------------
AUDIO_DIR   = Path(r"F:\CMU\25spring\10623genai\project\blip2_audioldm2_pair\audio")
IMAGE_DIR   = Path(r"F:\CMU\25spring\10623genai\project\blip2_audioldm2_pair\images")
MODEL_FILE  = Path("../assets/AudioCLIP-Full-Training.pt")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
BATCH_AUD   = 4
BATCH_IMG   = 16
SAMPLE_RATE = 44100
IMAGE_SIZE  = 224
CSV_OUT     = r"F:\CMU\25spring\10623genai\project\blip2_audioldm2_pair\similarity_results.csv"
# -------------------------------------

torch.set_grad_enabled(False)


print("Loading AudioCLIP …")
aclp = AudioCLIP(pretrained=str(MODEL_FILE)).to(DEVICE).eval()

audio_tf = ToTensor1D()
image_tf = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Resize(IMAGE_SIZE, interpolation=Image.BICUBIC),
    tv.transforms.CenterCrop(IMAGE_SIZE),
])


wav_paths = sorted(AUDIO_DIR.glob("*.wav"))
tracks = []
print("\nLoading audio files …")
for p in tqdm(wav_paths):
    track, _ = librosa.load(p, sr=SAMPLE_RATE, dtype=np.float32)
    tracks.append(track)


audio_tensors = [audio_tf(t.reshape(1, -1)) for t in tracks]


img_paths = sorted(IMAGE_DIR.glob("*.jpg"))
images_raw = []
print("\nLoading images …")
for p in tqdm(img_paths):
    images_raw.append(simplejpeg.decode_jpeg(p.read_bytes()))

image_tensors = [image_tf(img) for img in images_raw]


print("\nComputing audio features …")
audio_feats = []
for i in tqdm(range(0, len(audio_tensors), BATCH_AUD)):
    batch = torch.stack(audio_tensors[i:i+BATCH_AUD]).to(DEVICE, dtype=DTYPE, non_blocking=True)
    ((af, _, _), _), _ = aclp(audio=batch)
    audio_feats.append(af.float().cpu())          # 回收显存
    del batch
    torch.cuda.empty_cache()
audio_feat = torch.cat(audio_feats)


print("\nComputing image features …")
image_feats = []
for i in tqdm(range(0, len(image_tensors), BATCH_IMG)):
    batch = torch.stack(image_tensors[i:i+BATCH_IMG]).to(DEVICE, dtype=DTYPE, non_blocking=True)
    ((_, imf, _), _), _ = aclp(image=batch)
    image_feats.append(imf.float().cpu())
    del batch
    torch.cuda.empty_cache()
image_feat = torch.cat(image_feats)


audio_feat = audio_feat / audio_feat.norm(dim=-1, keepdim=True)
image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
scale = torch.clamp(aclp.logit_scale_ai.exp(), 1.0, 100.0).cpu()
logits = scale * audio_feat @ image_feat.T         # [num_audio, num_images]
#
# TOPK = 4
# print("\nImage                            Top-3 Audio (Sim)                                | Self-paired\n")
# for img_idx, img_path in enumerate(img_paths):
#     vals, ids = logits[:, img_idx].topk(TOPK)
#     topk_str = ', '.join([f'{wav_paths[i].name} ({v:.4f})' for v, i in zip(vals, ids)])
#
#     stem = img_path.stem
#     self_score = "N/A"
#     idx_match = next((j for j, p in enumerate(wav_paths) if p.stem == stem), None)
#     if idx_match is not None:
#         self_score = f"{logits[idx_match, img_idx]:.4f}"
#
#     print(f"{img_path.name:>30s} -> {topk_str:<70s} | {self_score}")
TOPK = 3
with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "top1", "top2", "top3", "self_score"])

    for img_idx, img_path in enumerate(img_paths):
        vals, ids = logits[:, img_idx].topk(TOPK)
        tops = [f"{wav_paths[i].name} ({v:.4f})" for v, i in zip(vals, ids)]

        stem = img_path.stem
        idx_match = next((j for j, p in enumerate(wav_paths) if p.stem == stem), None)
        self_score = f"{logits[idx_match, img_idx]:.4f}" if idx_match is not None else "N/A"

        while len(tops) < TOPK:
            tops.append("N/A")

        writer.writerow([img_path.name, *tops, self_score])

print(f"\nSaved at {CSV_OUT}")