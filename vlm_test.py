from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Select a tiny, CPU-friendly Vision Model
model_id = "vikhyatk/moondream2"
revision = "2024-08-26" # Locks in a stable, verified version

print("Loading model and tokenizer into CPU RAM (this may take a minute)...")
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

# 2. Load your local image file 
# Change 'test.jpg' to the actual path of an image on your computer!
image_path = "test.jpg" 
try:
    image = Image.open(image_path)
except FileNotFoundError:
    print(f"Error: Could not find an image at '{image_path}'. Please update the path in the script.")
    exit()

# 3. Encode the image using the model's built-in vision encoder
print("Processing image features...")
enc_image = model.encode_image(image)

# 4. Ask your question
prompt = "Describe what you see in this image in detail."
print(f"\nUser Prompt: {prompt}")
print("Generating response on Intel CPU...")

# 5. Run inference and print the result
response = model.answer_question(enc_image, prompt, tokenizer)
print("\nModel Response:")
print(response)
