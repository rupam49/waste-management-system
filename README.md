# ♻️ Edge AI Waste Segregation System (Hackathon PoC)

A real-time, 100% offline edge-optimized waste classification system built using PyTorch, Meta ExecuTorch, and Gradio. Designed for smart canteens, automated recycling units, and municipal waste sorting facilities with zero cloud dependency.

---

## 🚀 Overview & Problem Statement
Traditional waste sorting systems rely heavily on cloud-based APIs, introducing privacy risks, network dependency, and high latency. This project solves these challenges by compiling and running a lightweight computer vision model directly on edge hardware. Using **Meta ExecuTorch**, the model achieves ultra-low latency inference locally, making it ideal for deployment on edge devices like Raspberry Pi, Jetson Nano, or local hardware kiosks.

---

## 🛠️ Tech Stack & Architecture
- **Deep Learning Framework:** PyTorch & Torchvision (`MobileNetV3-Small`)
- **Edge Runtime:** Meta ExecuTorch (`.pte` format for cross-platform local execution)
- **Web Interface:** Gradio (Interactive local dashboard)
- **Environment:** Python / Windows Local Development (Target: Raspberry Pi OS)

---

## 📂 Project Structure
```text
edge_waste_sorting/
│
├── app.py                  # Real-time Gradio Web UI for live inference
├── export_model.py         # Model export and optimization script to ExecuTorch format
├── evaluate_waste.py       # Performance benchmarking script (Latency & Model size)
├── waste_classifier.pte    # Compiled lightweight ExecuTorch binary
├── imagenet_classes.txt    # Category label mapping for real-world classes
└── test_images/            # Sample test dataset for evaluation