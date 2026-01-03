"""
fire_smoke_multi_mode.py

Supports:
 - 1: Webcam (camera index 0)
 - 2: Pick an image file
 - 3: Pick a local video file
 - 4: CCTV / RTSP / HTTP stream URL

Features:
 - Uses ultralytics YOLO model (YOLOv8)
 - Saves annotated images and videos into fire_smoke_results/fire or /smoke
 - Logs detections into fire_smoke_results/fire_smoke_alerts_log.csv
 - FPS & top detection confidences overlay
 - Beep alert on detection (winsound on Windows, fallback otherwise)
"""

import os
import cv2
import time
import platform
from datetime import datetime
from tkinter import Tk, filedialog, simpledialog
from ultralytics import YOLO

# Try Windows beep, otherwise fallback
try:
    import winsound
    _HAS_WINSOUND = True
except Exception:
    _HAS_WINSOUND = False

# ----------------------------
# ========== CONFIG ==========
# ----------------------------
MODEL_PATH = r"C:\Users\singh\OneDrive\Desktop\RP CG\FirenSmoke\results\fire_smoke_cpu\yolov8n_v1\weights\best.pt"
CONFIDENCE_THRESHOLD = 0.2   # default threshold (tweakable)
BASE_SAVE_DIR = "fire_smoke_results"
LOG_FILE = os.path.join(BASE_SAVE_DIR, "fire_smoke_alerts_log.csv")
SAVE_VIDEO = True            # save annotated video outputs by default

# ----------------------------
# ========== SETUP ===========
# ----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at: {MODEL_PATH}")

os.makedirs(BASE_SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_SAVE_DIR, "fire"), exist_ok=True)
os.makedirs(os.path.join(BASE_SAVE_DIR, "smoke"), exist_ok=True)

# Ensure log file exists and has header
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as fh:
        fh.write("timestamp,label,confidence,source,filename\n")

print(f"[INFO] Loading model from:\n{MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("[INFO] Model loaded successfully ✅")
print(f"[INFO] Class names: {model.names}\n")

# Simple beep function
def beep_alert(duration=300, freq=1200):
    try:
        if _HAS_WINSOUND and platform.system() == "Windows":
            winsound.Beep(freq, duration)
        else:
            # fallback: system bell (may or may not sound depending on environment)
            print("\a", end="", flush=True)
    except Exception:
        pass

def log_alert(label, confidence, source, filename):
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"{time_now},{label},{confidence:.2f},{source},{filename}\n")

# Overlay helper
def draw_hud(frame, fps, top_texts):
    """
    top_texts: list of strings to display on top-left, stacked
    """
    x, y0 = 10, 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    # background box for readability
    overlay_h = 20 * (1 + len(top_texts))
    cv2.rectangle(frame, (5, 5), (330, 5 + overlay_h), (0,0,0), -1)
    # fps
    cv2.putText(frame, f"FPS: {fps:.1f}", (x, y0), font, font_scale, (0,255,0), thickness)
    # additional texts
    for i, t in enumerate(top_texts):
        cv2.putText(frame, t, (x, y0 + 22 * (i+1)), font, 0.55, (0,255,255), 1)

# Choose mode
print("Choose an option:")
print("1 - Run Webcam Detection (with alerts & save)")
print("2 - Upload an Image File")
print("3 - Upload a Local Video File")
print("4 - CCTV / RTSP / HTTP Stream (enter URL)")

choice = input("Enter your choice (1/2/3/4): ").strip()

# Generic function to process frame results and handle saving + logging
def process_results_and_alerts(results, annotated_frame, source_label, filename_hint, save_base_dir=BASE_SAVE_DIR):
    """
    results: ultralytics Results (for a single frame or image)
    annotated_frame: numpy image (annotated)
    source_label: string describing source (e.g., 'webcam', 'image', 'video', 'cctv')
    filename_hint: base name to use for saving snapshots
    Returns: list of detected (label, confidence) above threshold
    """
    detected = []
    if results and len(results) > 0:
        r = results[0]
        boxes = getattr(r, "boxes", [])
        for box in boxes:
            conf = float(box.conf)
            cls = int(box.cls)
            label = model.names[cls].lower()
            if conf >= CONFIDENCE_THRESHOLD:
                detected.append((label, conf))
                # Log & beep & save snapshot into subfolder
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                subfolder = os.path.join(save_base_dir, label)
                os.makedirs(subfolder, exist_ok=True)
                # filename includes hint, label and timestamp
                out_filename = f"{filename_hint}_{label}_{timestamp}.jpg"
                out_path = os.path.join(subfolder, out_filename)
                try:
                    cv2.imwrite(out_path, annotated_frame)
                    print(f"[INFO] Snapshot saved at: {out_path}")
                except Exception as e:
                    print(f"[WARN] Could not save snapshot: {e}")
                # beep and log
                beep_alert()
                log_alert(label, conf, source_label, out_filename)
    return detected

# ---------- MODE: WEBCAM ----------
if choice == "1":
    print("[INFO] Starting webcam detection... Press 'Q' to quit.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Unable to access webcam. Exiting.")
        exit()

    # Video writer setup (if saving)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    save_vid_path = os.path.join(BASE_SAVE_DIR, f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4") if SAVE_VIDEO else None
    out_writer = None
    if SAVE_VIDEO:
        # need fps and frame size - try to read from capture
        time.sleep(0.5)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 120:
            fps = 20.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_writer = cv2.VideoWriter(save_vid_path, fourcc, fps, (frame_w, frame_h))
        print(f"[INFO] Saving annotated webcam video to: {save_vid_path}")

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from webcam. Exiting loop.")
            break

        # Run model on frame (returns Results)
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, save=False)
        annotated = results[0].plot() if (results and len(results) > 0) else frame.copy()

        # Compute FPS
        cur_time = time.time()
        fps = 1.0 / (cur_time - prev_time) if (cur_time - prev_time) > 0 else 0.0
        prev_time = cur_time

        # Prepare top detection texts (take top 2 by confidence)
        top_texts = []
        if results and len(results) > 0:
            confs = []
            for box in results[0].boxes:
                confs.append(float(box.conf))
            if confs:
                # sort descending and pick top 2 confidences and their labels
                sorted_boxes = sorted(results[0].boxes, key=lambda b: float(b.conf), reverse=True)
                for b in sorted_boxes[:2]:
                    top_texts.append(f"{model.names[int(b.cls)].upper()} {float(b.conf):.2f}")

        draw_hud(annotated, fps, top_texts)

        # Process results: alert, save snapshot per detection
        process_results_and_alerts(results, annotated, "webcam", "webcam_frame")

        cv2.imshow("🔥 Fire & Smoke Detection - Webcam", annotated)

        # Write frame to video if enabled
        if out_writer is not None:
            # ensure frame size same as writer expect
            try:
                out_writer.write(annotated)
            except Exception:
                pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quitting webcam mode.")
            break

    cap.release()
    if out_writer:
        out_writer.release()
        print(f"[INFO] Saved webcam video to: {save_vid_path}")
    cv2.destroyAllWindows()

# ---------- MODE: IMAGE ----------
elif choice == "2":
    print("[INFO] Opening file picker for image...")
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select an Image",
                                           filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    root.destroy()
    if not file_path:
        print("❌ No file selected. Exiting.")
    else:
        print(f"[INFO] Running inference on: {file_path}")
        results = model.predict(file_path, conf=CONFIDENCE_THRESHOLD, save=False)
        annotated = results[0].plot() if (results and len(results) > 0) else cv2.imread(file_path)

        # draw HUD with fps=0 for image
        draw_hud(annotated, 0.0, [])

        # Process detection(s) and save snapshots in subfolders
        detected = process_results_and_alerts(results, annotated, "image", os.path.splitext(os.path.basename(file_path))[0])

        # Always save annotated image to a top-level image_results folder (use timestamp)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_img_name = f"image_{timestamp}.jpg"
        out_img_path = os.path.join(BASE_SAVE_DIR, out_img_name)
        try:
            cv2.imwrite(out_img_path, annotated)
            print(f"[INFO] Annotated result saved at: {out_img_path}")
        except Exception as e:
            print(f"[WARN] Could not save annotated image: {e}")

        cv2.imshow("🔥 Fire & Smoke Detection - Image", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if detected:
            print(f"[INFO] Detections: {detected}")
        else:
            print("[INFO] No detections above threshold.")

# ---------- MODE: LOCAL VIDEO ----------
elif choice == "3":
    print("[INFO] Opening file picker for video...")
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select a Video",
                                           filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")])
    root.destroy()
    if not file_path:
        print("❌ No file selected. Exiting.")
    else:
        print(f"[INFO] Running inference on video: {file_path}")
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print("❌ Unable to open video. Exiting.")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0 or fps > 120:
                fps = 20.0
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            save_vid_path = os.path.join(BASE_SAVE_DIR, f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            out_writer = cv2.VideoWriter(save_vid_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h))
            prev_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, save=False)
                annotated = results[0].plot() if (results and len(results) > 0) else frame.copy()

                # FPS
                cur_time = time.time()
                fps_cur = 1.0 / (cur_time - prev_time) if (cur_time - prev_time) > 0 else 0.0
                prev_time = cur_time

                # HUD
                top_texts = []
                if results and len(results) > 0:
                    sorted_boxes = sorted(results[0].boxes, key=lambda b: float(b.conf), reverse=True)
                    for b in sorted_boxes[:2]:
                        top_texts.append(f"{model.names[int(b.cls)].upper()} {float(b.conf):.2f}")
                draw_hud(annotated, fps_cur, top_texts)

                # Process detection alerts and snapshots
                process_results_and_alerts(results, annotated, "video", os.path.splitext(os.path.basename(file_path))[0])

                out_writer.write(annotated)
                cv2.imshow("🔥 Fire & Smoke Detection - Video", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] User requested quit.")
                    break

            cap.release()
            out_writer.release()
            cv2.destroyAllWindows()
            print(f"[INFO] Saved annotated video to: {save_vid_path}")

# ---------- MODE: CCTV / RTSP ----------
elif choice == "4":
    # Request URL via dialog for nicer UX
    root = Tk()
    root.withdraw()
    stream_url = simpledialog.askstring("CCTV Stream", "Enter RTSP/HTTP stream URL (e.g. rtsp://user:pass@ip:554/stream):")
    root.destroy()
    if not stream_url:
        print("❌ No stream URL entered. Exiting.")
    else:
        print(f"[INFO] Connecting to stream: {stream_url}")
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            print("❌ Unable to open stream. Exiting.")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0 or fps > 120:
                fps = 20.0
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            save_vid_path = os.path.join(BASE_SAVE_DIR, f"cctv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4") if SAVE_VIDEO else None
            out_writer = cv2.VideoWriter(save_vid_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h)) if SAVE_VIDEO else None
            print("[INFO] Stream opened. Press 'Q' in window to quit.")

            prev_time = time.time()
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[WARN] Failed to read frame from stream; retrying...")
                    time.sleep(0.5)
                    continue

                results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, save=False)
                annotated = results[0].plot() if (results and len(results) > 0) else frame.copy()

                # FPS
                cur_time = time.time()
                fps_cur = 1.0 / (cur_time - prev_time) if (cur_time - prev_time) > 0 else 0.0
                prev_time = cur_time

                # HUD
                top_texts = []
                if results and len(results) > 0:
                    sorted_boxes = sorted(results[0].boxes, key=lambda b: float(b.conf), reverse=True)
                    for b in sorted_boxes[:2]:
                        top_texts.append(f"{model.names[int(b.cls)].upper()} {float(b.conf):.2f}")
                draw_hud(annotated, fps_cur, top_texts)

                # process detections
                process_results_and_alerts(results, annotated, "cctv", "cctv_frame")

                # write to output video if enabled
                if out_writer:
                    out_writer.write(annotated)

                cv2.imshow("🔥 Fire & Smoke Detection - CCTV Stream", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] User requested quit.")
                    break

            cap.release()
            if out_writer:
                out_writer.release()
                print(f"[INFO] Saved annotated stream video to: {save_vid_path}")
            cv2.destroyAllWindows()

else:
    print("❌ Invalid choice. Run script again and pick 1/2/3/4.")
