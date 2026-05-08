import cv2
import json
import time
import numpy as np
import requests
from ultralytics import YOLO

# ----------------------------
# Configuration
# ----------------------------

# ----------------------------
# Night / Low Light Config
# ----------------------------

ENABLE_NIGHT_PROTECTION = True

# 0 = black, 255 = bright
# If ROI average brightness is below this, we consider it too dark
MIN_BRIGHTNESS_THRESHOLD = 35

# If True, detector keeps last known status during darkness
KEEP_LAST_STATUS_WHEN_DARK = True

CAMERA_INDEX = 0
ROI_FILE = "parking_rois.json"

MODEL_NAME = "yolov8n.pt"

CHECK_INTERVAL_SECONDS = 3
STATUS_CONFIRMATION_COUNT = 3

VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]

YOLO_CONFIDENCE = 0.25
OVERLAP_THRESHOLD = 0.05
PIXEL_CHANGE_THRESHOLD = 0.08

ENABLE_TELEGRAM = False

TELEGRAM_BOT_TOKEN = "PUT_YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "PUT_YOUR_CHAT_ID_HERE"

# ----------------------------
# Telegram Alert
# ----------------------------

def send_telegram_alert(message):
    if not ENABLE_TELEGRAM:
        print(f"[ALERT] {message}")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"Telegram alert failed: {e}")

# ----------------------------
# ROI Helpers
# ----------------------------

def load_rois():
    with open(ROI_FILE, "r") as f:
        return json.load(f)

def create_roi_mask(frame_shape, points):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    polygon = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    return mask

def bbox_to_mask(frame_shape, box):
    x1, y1, x2, y2 = map(int, box)
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask

def calculate_overlap(frame_shape, roi_points, box):
    roi_mask = create_roi_mask(frame_shape, roi_points)
    box_mask = bbox_to_mask(frame_shape, box)

    intersection = cv2.bitwise_and(roi_mask, box_mask)

    roi_area = cv2.countNonZero(roi_mask)
    intersection_area = cv2.countNonZero(intersection)

    if roi_area == 0:
        return 0

    return intersection_area / roi_area

def roi_pixel_change_score(current_gray, background_gray, roi_points):
    mask = create_roi_mask(current_gray.shape, roi_points)

    diff = cv2.absdiff(current_gray, background_gray)
    _, threshold = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)

    roi_diff = cv2.bitwise_and(threshold, threshold, mask=mask)

    changed_pixels = cv2.countNonZero(roi_diff)
    roi_pixels = cv2.countNonZero(mask)

    if roi_pixels == 0:
        return 0

    return changed_pixels / roi_pixels

# ----------------------------
# Background Calibration
# ----------------------------

def capture_background(cap):
    print("Capturing background reference...")
    print("IMPORTANT: Best result is when spaces are empty.")
    print("If spaces are not empty, system can still work but background confirmation is less accurate.")

    frames = []

    for _ in range(20):
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        time.sleep(0.1)

    if not frames:
        raise RuntimeError("Could not capture background frames.")

    background = np.median(frames, axis=0).astype(np.uint8)
    return background

# ----------------------------
# Main
# ----------------------------
def get_roi_brightness(frame_gray, roi_points):
    mask = create_roi_mask(frame_gray.shape, roi_points)
    roi_pixels = frame_gray[mask == 255]

    if roi_pixels.size == 0:
        return 0

    return float(np.mean(roi_pixels))


def main():
    rois = load_rois()

    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    background_gray = capture_background(cap)

    last_status = {}
    pending_status = {}
    pending_count = {}

    for roi in rois:
        last_status[roi["name"]] = "unknown"
        pending_status[roi["name"]] = "unknown"
        pending_count[roi["name"]] = 0

    print("Parking detector started.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera frame read failed.")
            time.sleep(2)
            continue

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = model(frame, verbose=False, conf=YOLO_CONFIDENCE)

        detected_boxes = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

                if class_name in VEHICLE_CLASSES:
                    xyxy = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    detected_boxes.append((xyxy, class_name, confidence))

        display = frame.copy()

        for roi in rois:
            space_name = roi["name"]
            roi_points = roi["points"]

            yolo_vehicle_present = False

            for box, class_name, confidence in detected_boxes:
                overlap = calculate_overlap(frame.shape, roi_points, box)

                if overlap >= OVERLAP_THRESHOLD:
                    yolo_vehicle_present = True

                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(
                        display,
                        f"{class_name} {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2
                    )

            pixel_score = roi_pixel_change_score(frame_gray, background_gray, roi_points)
            pixel_vehicle_present = pixel_score >= PIXEL_CHANGE_THRESHOLD

            roi_brightness = get_roi_brightness(frame_gray, roi_points)

            too_dark = (
            ENABLE_NIGHT_PROTECTION
            and roi_brightness < MIN_BRIGHTNESS_THRESHOLD
)

            if too_dark:
                if KEEP_LAST_STATUS_WHEN_DARK and last_status[space_name] != "unknown":
                    current_status = last_status[space_name]
                else:
                    current_status = "unknown"

                print(
                    f"{space_name}: LOW LIGHT detected. "
                    f"brightness={roi_brightness:.1f}. "
                    f"Keeping status as {current_status}"
    )

            else:
                if yolo_vehicle_present or pixel_vehicle_present:
                    current_status = "occupied"
                else:
                    current_status = "available"

            # Debounce logic to avoid alert flapping
            if current_status == pending_status[space_name]:
                pending_count[space_name] += 1
            else:
                pending_status[space_name] = current_status
                pending_count[space_name] = 1

            if pending_count[space_name] >= STATUS_CONFIRMATION_COUNT:
                    if current_status != "unknown" and last_status[space_name] != current_status:
                        last_status[space_name] = current_status
                

                    message = f"Parking {space_name} is now {current_status.upper()}"
                    send_telegram_alert(message)

            # Draw ROI
            pts = np.array(roi_points, np.int32)
            color = (0, 0, 255) if last_status[space_name] == "occupied" else (0, 255, 0)
            cv2.polylines(display, [pts], True, color, 3)

            label_position = tuple(roi_points[0])
            cv2.putText(
                display,
                f"{space_name}: {last_status[space_name]} pixel={pixel_score:.2f}",
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        cv2.imshow("Parking Detector", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        time.sleep(CHECK_INTERVAL_SECONDS)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
