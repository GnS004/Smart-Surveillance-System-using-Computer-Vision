# Garbage Detection Module

This module is part of the **Smart Surveillance System** and performs **real‑time garbage and waste detection** using a deep learning model. It uses a **custom dataset** specifically created and annotated for detecting different types of garbage, enabling more accurate performance in real‑world scenarios such as public spaces, streets, parks, and campuses.

---

## 🧠 Overview

The Garbage Detection Module uses a YOLO‑based object detection model trained on a **custom garbage dataset** to detect and locate garbage instances in images and video streams. It supports multiple input sources including images, video files, webcam feeds, and CCTV/RTSP camera streams.

This module is ideal for applications such as:
- Smart city cleanliness monitoring
- Environmental monitoring systems
- Public park and campus surveillance
- Automated waste management alerts

---

## 📁 Module Contents

| File | Description |
|------|-------------|
| `train.py` | Script to train the custom garbage detection model |
| `inference.py` | Script for running real‑time garbage detection |

---

## 🚀 Features

- Detects garbage and waste using a **custom‑trained deep learning model**
- Real‑time detection on:
  - 📸 Image files
  - 📼 Video files
  - 🎥 Webcam
  - 📡 CCTV / RTSP streams
- Bounding box visualization with confidence scores
- Lightweight and extendable

---

## 🛠️ Setup Instructions

1. Clone the repository and navigate to this folder:
   ```bash
   git clone https://github.com/GnS004/Smart-Surveillance-System-using-Computer-Vision.git
   cd Smart-Surveillance-System-using-Computer-Vision/garbage_detection

