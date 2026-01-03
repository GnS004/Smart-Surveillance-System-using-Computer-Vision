# Fire & Smoke Detection Module

This module is a computer vision system that performs **real‑time fire and smoke detection** using deep learning.  
It is part of the **Smart Surveillance System** and uses a **custom‑trained dataset** to identify fire and smoke accurately across diverse environments.

---

## 🧠 Overview

Fire & Smoke Detection helps monitor safety concerns in environments such as:

- Public spaces and campuses  
- Industrial and warehouse zones  
- Forest and outdoor surveillance  
- CCTV and security systems

The model uses a **custom dataset** annotated specifically for fire and smoke classes, trained with YOLO (You Only Look Once) family of object detection models to achieve high performance in real‑time applications.

---

## 📁 Module Contents

| File | Description |
|------|-------------|
| `train.py` | Script to train the custom fire & smoke detection model |
| `inference.py` | Script to run detection on images, videos, webcams, and streams |

---

## 🚀 Features

- Detects **fire** and **smoke** in real time  
- Built on **deep learning (YOLO)** for accurate object detection  
- Supports multiple input sources:
  - 📸 Image files
  - 📼 Video files
  - 🎥 Webcam
  - 📡 CCTV / RTSP streams  
- Uses **custom dataset** for stronger performance on fire and smoke scenarios

---

## 🛠️ Setup Instructions

1. **Clone the entire repository**
   ```bash
   git clone https://github.com/GnS004/Smart-Surveillance-System-using-Computer-Vision.git
   cd Smart-Surveillance-System-using-Computer-Vision/fire_smoke_detection

