import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import os

def load_model_and_processor(model_path):
    """Load the model and processor from the saved path."""
    processor = AutoProcessor.from_pretrained(model_path)
    model = Blip2ForConditionalGeneration.from_pretrained(model_path)
    return model, processor

def generate_caption(image_path, model, processor):
    """Generate a caption for the given image."""
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    
    # Process the image
    inputs = processor(images=image, return_tensors="pt")
    
    # Move inputs to the same device as the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate caption
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            max_length=50,
            num_beams=5,
            length_penalty=1.0,
            temperature=1.0
        )
    
    # Decode the generated caption
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption

def main():
    # Path to the saved model
    model_path = "./emo-captioning-model"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    # Load model and processor
    print("Loading model and processor...")
    model, processor = load_model_and_processor(model_path)
    
    # Example image path (using the first image from your dataset)
    image_path = "data/images/nm5514001_rm484898816_1984-2-6_2013.jpg"
    
    # Generate caption
    print(f"\nGenerating caption for image: {image_path}")
    caption = generate_caption(image_path, model, processor)
    
    print("\nGenerated Caption:")
    print(caption)

if __name__ == "__main__":
    main() 