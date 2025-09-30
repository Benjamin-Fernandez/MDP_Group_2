# app.py - RPi YOLO live stream + UART motor control via STM32
from flask import Flask, Response, render_template_string, send_from_directory
from picamera2 import Picamera2
from libcamera import controls
import cv2, time, atexit, signal, sys, os, serial
import numpy as np
from ultralytics import YOLO
import threading
from queue import Queue
from datetime import datetime

# ---------- Perf knobs ----------
os.environ["OMP_NUM_THREADS"] = "2"
cv2.setUseOptimized(True)
cv2.setNumThreads(2)

# ---------- Settings ----------
WIDTH, HEIGHT = 320, 240
FPS = 30
JPEG_QUALITY = 80
WEIGHTS_PATH = "/home/mdp-grp2/rpi-yolo-flask/weights/best.onnx"
SAVE_DIR = "pics"
GALLERY_PAGE_SIZE = 200

os.makedirs(SAVE_DIR, exist_ok=True)
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Weights not found at {WEIGHTS_PATH}")

# ---------- Serial Port Setup ----------
try:
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    print("✅ UART connected: /dev/ttyACM0")
except Exception as e:
    print(f"❌ UART init failed: {e}")
    ser = None

# ---------- Flask ----------
app = Flask(__name__)

# ---------- Label mapping ----------
NAMES = {
    0:"Bullseye", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9",
    10:"A", 11:"B", 12:"C", 13:"D", 14:"E", 15:"F", 16:"G", 17:"H",
    18:"S", 19:"T", 20:"U", 21:"V", 22:"W", 23:"X", 24:"Y", 25:"Z",
    26:"Up", 27:"Down", 28:"Right", 29:"Left", 30:"Stop"
}

# ---------- Camera setup ----------
picam2 = Picamera2()
video_config = picam2.create_video_configuration(
    main={"size": (WIDTH, HEIGHT), "format": "BGR888"},
    controls={"FrameRate": FPS},
)
picam2.configure(video_config)
picam2.start()
picam2.set_controls({
    "AwbEnable": True,
    "AwbMode": controls.AwbModeEnum.Auto,
    "AeEnable": True,
    "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.Off,
})
time.sleep(0.6)

# ---------- Cleanup ----------
def _cleanup_only():
    global detection_thread, running, picam2
    running = False
    try:
        if 'detection_thread' in globals() and detection_thread.is_alive():
            detection_thread.join(timeout=1)
    except: pass
    try: picam2.stop()
    except: pass
    try: picam2.close()
    except: pass
    try:
        if ser and ser.is_open:
            ser.close()
    except: pass

def _handle_signal(signum, frame):
    _cleanup_only()
    os._exit(0)

atexit.register(_cleanup_only)
signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ---------- Load YOLO model ----------
model = YOLO(WEIGHTS_PATH, task="detect")

# ---------- Threading ----------
frame_queue = Queue(maxsize=1)
detection_results = {"boxes": np.empty((0, 4), dtype=np.float32),
                     "confs": np.empty((0,), dtype=np.float32),
                     "clss":  np.empty((0,), dtype=np.int32)}
running = True

def _extract_results(ultra_result):
    if ultra_result is None or ultra_result.boxes is None or len(ultra_result.boxes) == 0:
        return (np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32))
    boxes = ultra_result.boxes.xyxy.cpu().numpy().astype(np.float32)
    confs = ultra_result.boxes.conf.cpu().numpy().astype(np.float32)
    clss  = ultra_result.boxes.cls.cpu().numpy().astype(np.int32)
    return boxes, confs, clss

def send_uart_command(char):
    if ser and ser.is_open:
        try:
            ser.write(char.encode())
        except Exception as e:
            print(f"UART send error: {e}")

# ---------- NEW: RFCOMM message sender ----------
def send_rfcomm_message(msg: str):
    try:
        os.system(f'echo -n "{msg}" > /dev/rfcomm0')
        print(f"📡 Sent to rfcomm0: {msg}")
    except Exception as e:
        print(f"RFCOMM send error: {e}")

def detection_worker():
    last_detection_time = 0.0
    DETECTION_INTERVAL = 1 / 12
    last_sent = None
    global detection_results

    while running:
        now = time.time()
        if now - last_detection_time < DETECTION_INTERVAL:
            time.sleep(0.005)
            continue
        if frame_queue.empty():
            time.sleep(0.005)
            continue
        try:
            frame = frame_queue.get_nowait()
            res = model(frame, imgsz=256, conf=0.45, iou=0.5, max_det=10, verbose=False, device="cpu")[0]
            boxes, confs, clss = _extract_results(res)
            detection_results = {"boxes": boxes, "confs": confs, "clss": clss}
            last_detection_time = now

            if len(clss) > 0:
                cls_names = [NAMES.get(int(cls), str(cls)) for cls in clss]
                if "Up" in cls_names and last_sent != "f":
                    send_uart_command("f"); last_sent = "f"
                elif "Down" in cls_names and last_sent != "b":
                    send_uart_command("b"); last_sent = "b"
                elif "Left" in cls_names and last_sent != "l":
                    send_uart_command("l"); last_sent = "l"
                elif "Right" in cls_names and last_sent != "r":
                    send_uart_command("r"); last_sent = "r"
                elif "Stop" in cls_names and last_sent != "s":
                    send_uart_command("s"); last_sent = "s"

        except Exception as e:
            print(f"Detection error: {e}")
            time.sleep(0.05)

detection_thread = threading.Thread(target=detection_worker, daemon=True)
detection_thread.start()

# ---------- HTML ----------
HTML = """
<!doctype html>
<html>
<head>
    <title>RPi Camera + YOLO (Dual Models)</title>
    <style>
        body { margin: 0; background: #1a1a1a; display: flex; flex-direction: column;
               justify-content: center; align-items: center; min-height: 100vh; font-family: Arial, sans-serif; }
        .container { text-align: center; padding: 16px; }
        .title { color: #00ff00; margin-bottom: 12px; font-size: 24px; text-shadow: 0 0 10px #00ff00; }
        img {
            max-width: 100%;
            width: 960px;
            height: auto;
            border: 2px solid #00ff00;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,255,0,0.3);
        }
        .status { color: #00ff00; margin-top: 10px; font-size: 14px; }
        .links a { color: #00ff00; margin: 0 10px; text-decoration: none; }
        .links { margin-bottom: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="title">🎯 Real-time Object Detection (Dual Models)</h1>
        <div class="links">
            <a href="/gallery">Open Gallery</a>
            <a href="/stats">Stats JSON</a>
        </div>
        <img src="/stream" alt="Camera Stream"/>
        <div class="status">● Live Detection Active</div>
    </div>
</body>
</html>
"""

GALLERY_HTML = """
<!doctype html>
<html>
<head>
    <title>Detections Gallery</title>
    <style>
        body { margin: 0; background: #111; color: #eee; font-family: Arial, sans-serif; }
        .wrap { max-width: 1100px; margin: 0 auto; padding: 16px; }
        h1 { color: #0f0; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }
        .card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; overflow: hidden; }
        .card img { width: 100%; height: 160px; object-fit: cover; display: block; }
        .meta { padding: 8px; font-size: 12px; color: #bbb; }
        a { color: #9f9; text-decoration: none; }
        .toplinks { margin-bottom: 12px; }
    </style>
</head>
<body>
<div class="wrap">
    <div class="toplinks"><a href="/">← Back to Live</a></div>
    <h1>Saved Detections</h1>
    <div class="grid">
        {% for f in files %}
        <div class="card">
            <a href="{{ url_for('serve_pic', filename=f) }}" target="_blank">
                <img src="{{ url_for('serve_pic', filename=f) }}" loading="lazy" alt="{{ f }}">
            </a>
            <div class="meta">{{ f }}</div>
        </div>
        {% endfor %}
    </div>
</div>
</body>
</html>
"""


# ---------- Routes ----------
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/pics/<path:filename>")
def serve_pic(filename):
    return send_from_directory(SAVE_DIR, filename)

@app.route("/gallery")
def gallery():
    try:
        files = [f for f in os.listdir(SAVE_DIR) if f.lower().endswith(".jpg")]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(SAVE_DIR, x)), reverse=True)
        files = files[:GALLERY_PAGE_SIZE]
    except Exception:
        files = []
    return render_template_string(GALLERY_HTML, files=files)


def mjpeg_generator():
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    last_seen_classes = set()

    while True:
        try:
            frame = picam2.capture_array()
            frame = frame[:, :, ::-1]
            frame = np.ascontiguousarray(frame)

            if not frame_queue.full():
                try:
                    frame_queue.put_nowait(frame.copy())
                except Exception:
                    pass

            h, w = frame.shape[:2]
            boxes = detection_results["boxes"]
            confs = detection_results["confs"]
            clss  = detection_results["clss"]

            detected_classes = set()
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                conf = confs[i]
                cls  = clss[i]

                x1 = max(0, min(int(x1), w-1))
                y1 = max(0, min(int(y1), h-1))
                x2 = max(0, min(int(x2), w-1))
                y2 = max(0, min(int(y2), h-1))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label_txt = f"{NAMES.get(cls, str(cls))} {conf:.2f}"
                label_size = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(frame,
                              (x1, max(0, y1 - label_size[1] - 6)),
                              (x1 + label_size[0], y1),
                              (0, 255, 0), -1)
                cv2.putText(frame, label_txt, (x1, max(16, y1 - 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                detected_classes.add(NAMES.get(cls, str(cls)))

            # Save & log only if the set of classes changed
            if len(detected_classes) > 0 and detected_classes != last_seen_classes:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"det_{ts}.jpg"
                out_path = os.path.join(SAVE_DIR, filename)
                classes_str = ", ".join(sorted(detected_classes))
                try:
                    cv2.imwrite(out_path, frame)
                    print(f"[{ts}] New detection: {classes_str} -> saved {out_path}")
                    send_rfcomm_message(f"Image Detected: {classes_str}")   # <-- added
                except Exception as e:
                    print(f"[{ts}] New detection: {classes_str} -> FAILED to save: {e}")
                last_seen_classes = detected_classes.copy()


            ok, buffer = cv2.imencode('.jpg', frame, encode_params)
            if not ok:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' +
                   frame_bytes + b'\r\n')

        except Exception as e:
            print(f"Stream error: {e}")
            time.sleep(0.1)


@app.route("/stream")
def stream():
    return Response(mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/stats")
def stats():
    return {
        "detections": int(len(detection_results["clss"])) ,
        "classes": [NAMES.get(int(cls), str(int(cls))) for cls in detection_results["clss"]],
        "confidences": detection_results["confs"].tolist() if len(detection_results["confs"]) > 0 else []
    }

# ---------- Flask startup ----------
if __name__ == "__main__":
    print("🚀 Starting optimized YOLO camera server...")
    print(f"📊 Model: {WEIGHTS_PATH}")
    print(f"🎥 Camera: {WIDTH}x{HEIGHT} @ {FPS}fps")
    print("🌐 Access at: http://your-rpi-ip:5000 (gallery at /gallery)")
    app.run(host="0.0.0.0", port=5001, threaded=True, debug=False, use_reloader=False)
