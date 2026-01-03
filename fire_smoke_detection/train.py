# train.py - Fire & Smoke Detection (YOLOv8 + CPU Friendly)
# Author: Hashy ⚡
# Description:
#   CPU-only training script for Fire & Smoke detection using YOLOv8 + Roboflow dataset.
#   Includes automatic validation and graph plotting after training.

import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from roboflow import Roboflow

# -----------------------------
# Step 1: Download Dataset
# -----------------------------
print("[INFO] Downloading dataset from Roboflow...")

rf = Roboflow(api_key="TnjvNfDV5ZfMaQWg59GA")
project = rf.workspace("campusguard").project("fire-and-smoke-ikpgn-pyemi")
version = project.version(1)
dataset = version.download("yolov8")

data_yaml = os.path.join(dataset.location, "data.yaml")
print(f"[INFO] Dataset ready at: {dataset.location}")
print(f"[INFO] Data YAML path: {data_yaml}")

# -----------------------------
# Step 2: Load YOLOv8 model
# -----------------------------
print("[INFO] Loading YOLOv8n (nano) model for CPU training...")
model = YOLO("yolov8n.pt")  # Lightest model, best for CPU training

# -----------------------------
# Step 3: Train Model
# -----------------------------
print("[INFO] Starting training... please wait 🧠")

results = model.train(
    data=data_yaml,
    epochs=25,                     # suitable for CPU
    imgsz=416,                     # smaller image size = faster
    batch=4,                       # small batch to avoid overload
    workers=2,                     # fewer workers for CPU
    device="cpu",                  # force CPU
    optimizer="SGD",               # simple optimizer
    lr0=0.01,
    patience=5,                    # early stopping
    project="results/fire_smoke_cpu",
    name="yolov8n_v1",
    verbose=True
)

print("\n[INFO] Training complete ✅")
print(f"[INFO] Results saved in: {results.save_dir}")

# -----------------------------
# Step 4: Validate the Model
# -----------------------------
print("\n[INFO] Running validation on best model weights...")

best_weights = os.path.join(results.save_dir, "weights", "best.pt")

if os.path.exists(best_weights):
    best_model = YOLO(best_weights)
    metrics = best_model.val()
    print("\n✅ Validation Complete!")
    print(metrics)
else:
    print("⚠️ No best.pt found. Skipping validation.")
    metrics = None

# -----------------------------
# Step 5: Plot Training Graphs
# -----------------------------
print("\n[INFO] Generating training performance graphs...")

results_dir = results.save_dir
metrics_file = os.path.join(results_dir, "results.csv")

if os.path.exists(metrics_file):
    import pandas as pd
    df = pd.read_csv(metrics_file)

    plt.figure(figsize=(12, 6))
    plt.suptitle("🔥 YOLOv8 Fire & Smoke Detection Training Results", fontsize=14)

    # Loss graph + mAP
    plt.subplot(1, 2, 1)
    plt.plot(df["epoch"], df["train/box_loss"], label="Box Loss")
    plt.plot(df["epoch"], df["train/cls_loss"], label="Class Loss")
    if "metrics/mAP50" in df.columns:
        plt.plot(df["epoch"], df["metrics/mAP50"], label="mAP@50")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Loss & mAP Progress")
    plt.legend()
    plt.grid(True)

    # Precision-Recall graph
    plt.subplot(1, 2, 2)
    if "metrics/precision(B)" in df.columns and "metrics/recall(B)" in df.columns:
        plt.plot(df["epoch"], df["metrics/precision(B)"], label="Precision")
        plt.plot(df["epoch"], df["metrics/recall(B)"], label="Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Precision vs Recall Over Epochs")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save graph as PNG
    plot_path = os.path.join(results_dir, "training_graphs.png")
    plt.savefig(plot_path)
    plt.show()

    print(f"[INFO] Graphs saved at: {plot_path}")
else:
    print("⚠️ results.csv not found. Could not plot graphs.")

# -----------------------------
# Wrap Up
# -----------------------------
print("\n🏁 All done boss! Best weights and graphs are ready.")
print(f"📁 Check folder: {results_dir}")
print(f"🥇 Best weights: {best_weights}")
