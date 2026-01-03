# train.py - Garbage Detection (YOLOv8 + CPU Friendly + Graphs)

import os
from ultralytics import YOLO
from roboflow import Roboflow
import matplotlib.pyplot as plt

print("[INFO] Downloading dataset from Roboflow...")
rf = Roboflow(api_key="TnjvNfDV5ZfMaQWg59GA")
project = rf.workspace("campusguard").project("garbage-detection-sozy9-p2l3p")
version = project.version(1)
dataset = version.download("yolov8")

data_yaml = os.path.join(dataset.location, "data.yaml")
print(f"[INFO] Dataset ready at: {dataset.location}")

# Load YOLOv8 nano model (smallest and fastest)
model = YOLO("yolov8n.pt")

print("[INFO] Starting training on CPU...")
results = model.train(
    data=data_yaml,
    epochs=50,
    imgsz=416,
    batch=4,
    workers=2,
    device="cpu",
    optimizer="SGD",
    lr0=0.01,
    patience=5,
    project="results/garbage_detection_cpu",
    name="yolov8n_v1",
    verbose=True
)

print("[INFO] Training complete!")
print(f"[INFO] Best weights saved at: {results.save_dir}/weights/best.pt")

# ===========================
# 📊 Training Visualization
# ===========================
try:
    print("[INFO] Generating training performance graphs...")
    metrics = results.results_dict

    plt.figure(figsize=(10, 6))
    plt.suptitle("Training Performance Overview", fontsize=14, fontweight='bold')

    # 1️⃣ Loss Graph
    plt.subplot(2, 2, 1)
    if 'train/loss' in metrics and 'val/loss' in metrics:
        plt.plot(metrics['train/loss'], label='Train Loss', color='blue')
        plt.plot(metrics['val/loss'], label='Val Loss', color='orange')
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
    else:
        plt.text(0.3, 0.5, "Loss Data Unavailable", fontsize=10)

    # 2️⃣ Precision Graph
    plt.subplot(2, 2, 2)
    if 'metrics/precision(B)' in metrics:
        plt.plot(metrics['metrics/precision(B)'], color='green')
        plt.title("Precision")
        plt.xlabel("Epochs")
        plt.ylabel("Precision")
    else:
        plt.text(0.3, 0.5, "Precision Data Unavailable", fontsize=10)

    # 3️⃣ Recall Graph
    plt.subplot(2, 2, 3)
    if 'metrics/recall(B)' in metrics:
        plt.plot(metrics['metrics/recall(B)'], color='red')
        plt.title("Recall")
        plt.xlabel("Epochs")
        plt.ylabel("Recall")
    else:
        plt.text(0.3, 0.5, "Recall Data Unavailable", fontsize=10)

    # 4️⃣ mAP Graph
    plt.subplot(2, 2, 4)
    if 'metrics/mAP50(B)' in metrics:
        plt.plot(metrics['metrics/mAP50(B)'], color='purple')
        plt.title("mAP@50")
        plt.xlabel("Epochs")
        plt.ylabel("mAP@50")
    else:
        plt.text(0.3, 0.5, "mAP Data Unavailable", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

except Exception as e:
    print(f"[WARN] Could not generate graphs: {e}")
