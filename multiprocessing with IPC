import cv2
import numpy as np
from urllib.parse import quote_plus
from multiprocessing import Process, Value
from ultralytics import YOLO
import time

FRAME_WIDTH, FRAME_HEIGHT = 640, 360

# -------------------------------
# Camera URL builders
# -------------------------------
def make_cpplus_rtsp_url(ip, username, password):
    return f"rtsp://{username}:{password}@{ip}:554/cam/realmonitor?channel=1&subtype=0"

def make_ipcam_rtsp_url(ip, username, password, path="stream1", port=554):
    u = quote_plus(username)
    p = quote_plus(password)
    return f"rtsp://{u}:{p}@{ip}:{port}/{path}"

# -------------------------------
# Human detection process
# -------------------------------
def human_detection_process(url, stop_flag, human_count, window_name):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"[{window_name}] Cannot open camera")
        return

    model = YOLO("yolov8n.pt")  # CPU-friendly small model

    while not stop_flag.value:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        results = model(frame, classes=[0])  # class 0 = person
        count = 0
        for r in results:
            boxes = r.boxes
            count += len(boxes)

        # Update shared human count
        human_count.value = count

        annotated = results[0].plot()
        cv2.imshow(window_name, annotated)

        key = cv2.waitKey(1) & 0xFF
        if window_name == "CP Plus" and key == ord('q'):
            stop_flag.value = 1
        elif window_name == "IP Camera" and key == ord('w'):
            stop_flag.value = 1

    cap.release()
    cv2.destroyWindow(window_name)
    print(f"[{window_name}] Process stopped")

# -------------------------------
# Comparison process with display
# -------------------------------
def compare_humans_with_display(counts, stop_flags):
    cp_count, ip_count = counts  # unpack counts

    while True:
        if stop_flags[0].value == 1 and stop_flags[1].value == 1:
            break

        cp = cp_count.value
        ip = ip_count.value

        # Create black display
        display = 255 * np.ones((200, 500, 3), dtype=np.uint8)

        # Determine which camera has more humans
        if cp > ip:
            text = f"More humans in CP Plus: {cp}"
        elif ip > cp:
            text = f"More humans in IP Camera: {ip}"
        else:
            text = f"Equal humans: {cp}"

        cv2.putText(display, text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Human Count Comparison", display)
        if cv2.waitKey(1000) & 0xFF == ord('e'):  # optional early exit
            break

    cv2.destroyWindow("Human Count Comparison")
    print("[Comparison] Process stopped")

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    # Shared stop flags
    cpplus_stop = Value('i', 0)
    ipcam_stop = Value('i', 0)

    # Shared human count variables
    cpplus_count = Value('i', 0)
    ipcam_count = Value('i', 0)

    # Camera URLs
    cpplus_url = make_cpplus_rtsp_url("192.168.31.70", "admin", "Mypassword@25")
    ipcam_url = make_ipcam_rtsp_url("192.168.31.68", "admin", "Mypassword@25")

    # Create human detection processes
    p1 = Process(target=human_detection_process, args=(cpplus_url, cpplus_stop, cpplus_count, "CP Plus"))
    p2 = Process(target=human_detection_process, args=(ipcam_url, ipcam_stop, ipcam_count, "IP Camera"))

    # Create comparison process
    p3 = Process(target=compare_humans_with_display,
                 args=([cpplus_count, ipcam_count], [cpplus_stop, ipcam_stop]))

    # Start processes
    p1.start()
    p2.start()
    p3.start()

    # Wait for all processes to finish
    p1.join()
    p2.join()
    p3.join()

    print("All processes stopped. Main program exiting.")
