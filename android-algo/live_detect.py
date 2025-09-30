import os, time, requests, tempfile
from picamera2 import Picamera2

PC_IP = "192.168.2.1"   # Your PC IP
OBSTACLE_ID = "01"
SIGNAL = "C"

def capture_to(path: str):
    # Initialize and use Picamera2
    picam2 = Picamera2()
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.start()
    time.sleep(1)  # Allow camera to warm up
    picam2.capture_file(path)
    picam2.close()

def post_image(img_path: str):
    url = f"http://{PC_IP}:5000/image"
    with open(img_path, "rb") as f:
        r = requests.post(url, files={"file": f}, timeout=30)
    r.raise_for_status()
    return r.json()

def main():
    print(f"Posting to http://{PC_IP}:5000/image. Ctrl+C to stop.")
    while True:
        with tempfile.TemporaryDirectory() as td:
            raw = os.path.join(td, "raw.jpg")
            capture_to(raw)
            ts = str(int(time.time()))
            api_name = os.path.join(td, f"{ts}_{OBSTACLE_ID}_{SIGNAL}.jpg")
            os.replace(raw, api_name)
            try:
                resp = post_image(api_name)
                print("Detection:", resp)
            except Exception as e:
                print("Upload failed:", e)
        time.sleep(0.8)

if __name__ == "__main__":
    main()
