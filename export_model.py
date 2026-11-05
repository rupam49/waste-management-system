import torch
import torchvision.models as models
from torch.export import export
from executorch.exir import to_edge

print("Loading pre-trained MobileNetV3-Small model...")
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
model.eval()

# Dummy input tensor (Standard input size for MobileNet: 1x3x224x224)
example_inputs = (torch.randn(1, 3, 224, 224),)

print("Exporting model using torch.export and lowering to ExecuTorch...")
torch_exported = export(model, example_inputs)
edge_program = to_edge(torch_exported)
executorch_program = edge_program.to_executorch()

# Yahan par .buffer ke aage se brackets () hata diye hain
with open("waste_classifier.pte", "wb") as f:
    f.write(executorch_program.buffer)

print("Success! Model saved as 'waste_classifier.pte'")