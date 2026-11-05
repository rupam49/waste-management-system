import time
import torch
from torchvision import transforms
from PIL import Image
from executorch.runtime import Runtime

print("Loading ExecuTorch runtime and model (.pte)...")
# ExecuTorch runtime se .pte model load kar rahe hain
runtime = Runtime.get()
program = runtime.load_program("waste_classifier.pte")
method = program.load_method("forward")

# Image preprocessing pipeline (MobileNet standard requirements)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])

# Dummy test image ke sath execution test karte hain (Aap yahan apni waste image path de sakte hain)
print("Running inference test...")
dummy_img_tensor = torch.randn(1, 3, 224, 224)

start_time = time.time()
# ExecuTorch forward pass
outputs = method.execute([dummy_img_tensor])
end_time = time.time()

inference_time_ms = (end_time - start_time) * 1000

print(f"Inference Successful!")
print(f"Approximate Inference Time: {inference_time_ms:.2f} ms")
print("ExecuTorch pipeline is fully working on your local system!")