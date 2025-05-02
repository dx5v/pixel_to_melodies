import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch
import shutil
import pynvml

def print_gpu_utilization():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024//1024}MB.")
    print(f"GPU memory total: {info.total//1024//1024}MB.")

cache_dir = os.path.expanduser("~/.cache/huggingface")
if os.path.exists(cache_dir):
    print(f"Clearing cache directory: {cache_dir}")
    shutil.rmtree(cache_dir)
    os.makedirs(cache_dir)

class ImageCaptioningDataset(Dataset):
    def __init__(self, csv_path, images_dir, processor):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        image_path = os.path.join(self.images_dir, item['image_name'])
        image = Image.open(image_path).convert('RGB')
        
        encoding = self.processor(images=image, padding="max_length", return_tensors="pt")

        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = item["annotation"]
        return encoding

def collate_fn(batch):
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch

quant_config = BitsAndBytesConfig(load_in_8bit=True)
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("ybelkada/blip2-opt-2.7b-fp16-sharded", device_map="auto", quantization_config=quant_config)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj"]
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

train_dataset = ImageCaptioningDataset("data/emo-at-cap.csv", "data/images", processor)
train_dataloader = DataLoader(
    train_dataset, 
    shuffle=True, 
    batch_size=8,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print_gpu_utilization()

total_epochs = 30
patience = 5
best_loss = float('inf')
patience_counter = 0


model.train()
gradient_accumulation_steps = 4

for epoch in range(total_epochs):
    print(f"\nEpoch {epoch + 1}/{total_epochs}")
    total_batches = len(train_dataloader)
    optimizer.zero_grad()
    epoch_loss = 0.0
    
    for idx, batch in enumerate(train_dataloader):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device, torch.float16)

        outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=input_ids)
        
        loss = outputs.loss / gradient_accumulation_steps
        epoch_loss += loss.item() * gradient_accumulation_steps
        print(f"Epoch {epoch + 1}/{total_epochs} - Batch {idx + 1}/{total_batches} - Loss: {loss.item() * gradient_accumulation_steps:.4f}")
        
        if device == "cuda" and idx % 10 == 0:
            print_gpu_utilization()

        loss.backward()
        
        if (idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    avg_epoch_loss = epoch_loss / len(train_dataloader)
    print(f"\nEpoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")

    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        patience_counter = 0
        model.save_pretrained("./best_model")
        processor.save_pretrained("./best_model")
        print(f"New best model saved with loss: {best_loss:.4f}")
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter} epochs")

    if patience_counter >= patience:
        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
        break

    if (epoch + 1) % 5 == 0:
        checkpoint_path = f"./checkpoints/epoch_{epoch+1}"
        os.makedirs("./checkpoints", exist_ok=True)
        model.save_pretrained(checkpoint_path)
        processor.save_pretrained(checkpoint_path)
        print(f"\nSaved checkpoint at epoch {epoch + 1}")

model.save_pretrained("./emo-captioning-model")
processor.save_pretrained("./emo-captioning-model")
print("\nTraining completed! Final model saved.")

