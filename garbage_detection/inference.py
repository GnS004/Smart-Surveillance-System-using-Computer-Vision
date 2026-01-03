import os
import cv2
from ultralytics import YOLO
from tkinter import Tk, filedialog
from datetime import datetime
import winsound  # For sound alert (Windows only)

# =============================
# 🧠 Model Path
# =============================
MODEL_PATH = r"C:\Users\singh\OneDrive\Desktop\RP CG\GarbageDetection\results\garbage_detection_cpu\yolov8n_v1\weights\best.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at: {MODEL_PATH}")

print(f"[INFO] Loading model from:\n{MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("[INFO] Model loaded successfully ✅\n")

# =============================
# ⚙️ Choose Mode
# =============================
print("Choose an option:")
print("1 - Run Webcam Detection (with alerts)")
print("2 - Upload an Image File")

choice = input("Enter your choice (1/2): ").strip()

# =============================
# 🚨 Parameters
# =============================
CONFIDENCE_THRESHOLD = 0.7
LOG_FILE = "garbage_alerts_log.csv"

def log_alert(label, confidence):
    """Append alert info to log file"""
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"{time_now},{label},{confidence:.2f}\n")

if choice == "1":
    print("[INFO] Starting webcam detection... Press 'Q' to quit.")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated = results[0].plot()

        # Check detections
        for box in results[0].boxes:
            conf = float(box.conf)
            cls = int(box.cls)
            label = model.names[cls]

            if conf > CONFIDENCE_THRESHOLD:
                print(f"🚨 ALERT: {label.upper()} detected! Confidence: {conf:.2f}")
                winsound.Beep(1000, 300)  # Beep sound for alert
                log_alert(label, conf)

        cv2.imshow("Garbage Detection - Webcam", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

elif choice == "2":
    print("[INFO] Opening file picker...")
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    root.destroy()

    if not file_path:
        print("❌ No file selected. Exiting.")
    else:
        print(f"[INFO] Running inference on: {file_path}")
        results = model(file_path)
        out_img = results[0].plot()

        for box in results[0].boxes:
            conf = float(box.conf)
            cls = int(box.cls)
            label = model.names[cls]

            if conf > CONFIDENCE_THRESHOLD:
                print(f"🚨 ALERT: {label.upper()} detected in image! Confidence: {conf:.2f}")
                winsound.Beep(1000, 300)
                log_alert(label, conf)

        cv2.imshow("Garbage Detection - Image", out_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f"[INFO] Results saved at: {results[0].save_dir}")

else:
    print("❌ Invalid choice. Please run again.")
