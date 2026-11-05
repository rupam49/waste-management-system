import time
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from executorch.runtime import Runtime
import urllib.request
import json
import gradio as gr

print("Loading ExecuTorch model and ImageNet labels...")
runtime = Runtime.get()
program = runtime.load_program("waste_classifier.pte")
method = program.load_method("forward")

# Download standard ImageNet class labels for realistic mapping
LABEL_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
try:
    urllib.request.urlretrieve(LABEL_URL, "imagenet_classes.txt")
except:
    pass

with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_waste(image):
    if image is None:
        return "Please upload an image.", "0.00 ms", "No Image"
    
    start_time = time.time()
    
    # Preprocess image and run inference
    input_tensor = preprocess(image).unsqueeze(0)
    outputs = method.execute([input_tensor])
    
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000
    
    # Get top prediction from model output tensor
    output_tensor = outputs[0]
    probabilities = torch.nn.functional.softmax(output_tensor[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)
    
    predicted_label = categories[top_catid[0].item()]
    confidence = top_prob[0].item() * 100
    
    # Smart Mapping: Real-world waste categorization based on detected object
    label_lower = predicted_label.lower()
    
    compost_keywords = ['fruit', 'vegetable', 'banana', 'apple', 'orange', 'food', 'bread', 'plate', 'cup', 'pizza', 'sandwich']
    recyclable_keywords = ['bottle', 'can', 'carton', 'paper', 'box', 'plastic', 'glass', 'cup', 'container', 'tin']
    
    if any(kw in label_lower for kw in compost_keywords):
        waste_category = "🟢 COMPOSTABLE (Organic/Food Waste)"
    elif any(kw in label_lower for kw in recyclable_keywords):
        waste_category = "🔵 DRY RECYCLABLE (Plastic/Metal/Paper)"
    else:
        waste_category = "🟡 GENERAL / LANDFILL WASTE"
        
    result_text = f"Detected Object: {predicted_label} ({confidence:.1f}% confidence)"
    
    return waste_category, result_text, f"{inference_time:.2f} ms"

# Gradio Web Interface design
demo = gr.Interface(
    fn=classify_waste,
    inputs=gr.Image(type="pil", label="Upload Real Waste Image"),
    outputs=[
        gr.Textbox(label="Segregation Decision"),
        gr.Textbox(label="Model Detection Details"),
        gr.Textbox(label="Edge Inference Latency")
    ],
    title="Smart Canteen Waste Segregator (Live Edge AI)",
    description="Real-time inference using PyTorch ExecuTorch. Upload a photo of a bottle, food item, or container to test."
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)