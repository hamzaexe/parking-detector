
import cv2
import json
import time
import numpy as np
import smtplib
from email.message import EmailMessage
from datetime import datetime
from ultralytics import YOLO

# ==========================================================
# Camera / Detection Config
# ==========================================================

CAMERA_INDEX = 0
ROI_FILE = "parking_rois.json"
MODEL_NAME = "yolov8n.pt"

CHECK_INTERVAL_SECONDS = 5
STATUS_CONFIRMATION_COUNT = 3

VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]

YOLO_CONFIDENCE = 0.25
OVERLAP_THRESHOLD = 0.05

SHOW_WINDOW = True

# ==========================================================
# Night / Low Light Protection
# ==========================================================

ENABLE_NIGHT_PROTECTION = True

# Increase this if it still says available in dark conditions.
MIN_BRIGHTNESS_THRESHOLD = 75

# At night, never mark as available.
DARK_STATUS = "occupied"

# ==========================================================
# Pixel Detection Config
# ==========================================================

ENABLE_PIXEL_CONFIRMATION = False
PIXEL_CHANGE_THRESHOLD = 0.10

# ==========================================================
# Email Alert Config
# ==========================================================

ENABLE_EMAIL = False

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

SMTP_USERNAME = "your_email@gmail.com"
SMTP_PASSWORD = "your_app_password"

EMAIL_FROM = "your_email@gmail.com"
EMAIL_TO = "destination_email@example.com"

EMAIL_SUBJECT_PREFIX = "[Parking Alert]"

ALERT_COOLDOWN_SECONDS = 300

# ==========================================================
# Email Function
# ==========================================================

def send_email_alert(subject, body):
    if not ENABLE_EMAIL:
        print(f"[ALERT] {subject}")
        print(body)
        return

    try:
        msg = EmailMessage()
        msg["From"] = EMAIL_FROM
        msg["To"] = EMAIL_TO
        msg["Subject"] = subject
        msg.set_content(body)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)

        print(f"[EMAIL SENT] {subject}")

    except Exception as e:
        print(f"[EMAIL FAILED] {e}")


# ==========================================================
# ROI Helpers
# ==========================================================

def load_rois():
    with open(ROI_FILE, "r") as f:
        rois = json.load(f)

    if not rois:
        raise RuntimeError("parking_rois.json is empty. Run calibrate_rois.py first.")

    print(f"Loaded {len(rois)} parking spaces from {ROI_FILE}")

    for roi in rois:
        print(f"ROI loaded: {roi['name']} - {roi['points']}")

    return rois


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
        return 0.0

    return intersection_area / roi_area


def get_roi_brightness(frame_gray, roi_points):
    mask = create_roi_mask(frame_gray.shape, roi_points)
    roi_pixels = frame_gray[mask == 255]

    if roi_pixels.size == 0:
        return 0.0

    return float(np.mean(roi_pixels))


def roi_pixel_change_score(current_gray, background_gray, roi_points):
    mask = create_roi_mask(current_gray.shape, roi_points)

    diff = cv2.absdiff(current_gray, background_gray)
    _, threshold = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)

    roi_diff = cv2.bitwise_and(threshold, threshold, mask=mask)

    changed_pixels = cv2.countNonZero(roi_diff)
    roi_pixels = cv2.countNonZero(mask)

    if roi_pixels == 0:
        return 0.0

    return changed_pixels / roi_pixels


# ==========================================================
# Background Capture
# ==========================================================

def capture_background(cap):
    print("Capturing background reference...")

    frames = []

    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        time.sleep(0.1)

    if not frames:
        raise RuntimeError("Could not capture background frames from camera.")

    background = np.median(frames, axis=0).astype(np.uint8)
    print("Background captured.")
    return background


# ==========================================================
# Status Decision Logic
# ==========================================================

def decide_status(frame_shape, frame_gray, roi_points, detected_boxes, background_gray):
    yolo_vehicle_present = False
    max_overlap = 0.0

    for box, class_name, confidence in detected_boxes:
        overlap = calculate_overlap(frame_shape, roi_points, box)
        max_overlap = max(max_overlap, overlap)

        if overlap >= OVERLAP_THRESHOLD:
            yolo_vehicle_present = True

    roi_brightness = get_roi_brightness(frame_gray, roi_points)

    too_dark = ENABLE_NIGHT_PROTECTION and roi_brightness < MIN_BRIGHTNESS_THRESHOLD

    pixel_score = 0.0
    pixel_vehicle_present = False

    if ENABLE_PIXEL_CONFIRMATION and background_gray is not None:
        pixel_score = roi_pixel_change_score(frame_gray, background_gray, roi_points)
        pixel_vehicle_present = pixel_score >= PIXEL_CHANGE_THRESHOLD

    if too_dark:
        current_status = DARK_STATUS
        reason = f"LOW_LIGHT_FAILSAFE_BRIGHTNESS_{roi_brightness:.1f}"
    elif yolo_vehicle_present:
        current_status = "occupied"
        reason = "YOLO_VEHICLE_DETECTED"
    elif pixel_vehicle_present:
        current_status = "occupied"
        reason = "PIXEL_CHANGE_DETECTED"
    else:
        current_status = "available"
        reason = "NO_VEHICLE_DETECTED"

    return {
        "status": current_status,
        "reason": reason,
        "brightness": roi_brightness,
        "too_dark": too_dark,
        "yolo_vehicle_present": yolo_vehicle_present,
        "max_overlap": max_overlap,
        "pixel_score": pixel_score,
        "pixel_vehicle_present": pixel_vehicle_present
    }


# ==========================================================
# Main Detector
# ==========================================================

def main():
    rois = load_rois()

    print("Loading YOLO model...")
    model = YOLO(MODEL_NAME)
    print("YOLO model loaded.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check CAMERA_INDEX or /dev/video0.")

    background_gray = None

    if ENABLE_PIXEL_CONFIRMATION:
        background_gray = capture_background(cap)

    last_confirmed_status = {}
    pending_status = {}
    pending_count = {}
    last_alert_time = {}

    for roi in rois:
        space_name = roi["name"]

        # Start as occupied to avoid false available at night startup.
        last_confirmed_status[space_name] = "occupied"
        pending_status[space_name] = "occupied"
        pending_count[space_name] = 0
        last_alert_time[space_name] = 0

    print("Parking detector started.")
    print(f"Night protection enabled: {ENABLE_NIGHT_PROTECTION}")
    print(f"Minimum brightness threshold: {MIN_BRIGHTNESS_THRESHOLD}")
    print(f"Dark status: {DARK_STATUS}")

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

        print("----- Detection cycle -----")
        print(f"YOLO vehicle boxes found: {len(detected_boxes)}")

        for roi in rois:
            space_name = roi["name"]
            roi_points = roi["points"]

            decision = decide_status(
                frame_shape=frame.shape,
                frame_gray=frame_gray,
                roi_points=roi_points,
                detected_boxes=detected_boxes,
                background_gray=background_gray
            )

            current_status = decision["status"]

            # Correct debounce logic
            if current_status == pending_status[space_name]:
                pending_count[space_name] += 1
            else:
                pending_status[space_name] = current_status
                pending_count[space_name] = 1

            if pending_count[space_name] >= STATUS_CONFIRMATION_COUNT:
                if last_confirmed_status[space_name] != current_status:
                    old_status = last_confirmed_status[space_name]
                    last_confirmed_status[space_name] = current_status

                    now = time.time()

                    if now - last_alert_time[space_name] >= ALERT_COOLDOWN_SECONDS:
                        last_alert_time[space_name] = now

                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        subject = f"{EMAIL_SUBJECT_PREFIX} {space_name} is {current_status.upper()}"

                        body = f"""
Parking status changed.

Space: {space_name}
Old status: {old_status.upper()}
New status: {current_status.upper()}
Time: {timestamp}

Reason: {decision["reason"]}
Brightness: {decision["brightness"]:.1f}
Too dark: {decision["too_dark"]}
YOLO vehicle detected: {decision["yolo_vehicle_present"]}
Max YOLO overlap: {decision["max_overlap"]:.2f}
Pixel score: {decision["pixel_score"]:.2f}

Camera: Logitech 720p on Raspberry Pi 4
"""

                        send_email_alert(subject, body)
                    else:
                        print(f"[NO ALERT] Cooldown active for {space_name}")

            confirmed_status = last_confirmed_status[space_name]

            print(
                f"{space_name}: "
                f"current={current_status}, "
                f"confirmed={confirmed_status}, "
                f"pending={pending_status[space_name]}, "
                f"pending_count={pending_count[space_name]}, "
                f"reason={decision['reason']}, "
                f"brightness={decision['brightness']:.1f}, "
                f"too_dark={decision['too_dark']}, "
                f"yolo={decision['yolo_vehicle_present']}, "
                f"overlap={decision['max_overlap']:.2f}, "
                f"pixel={decision['pixel_score']:.2f}"
            )

            pts = np.array(roi_points, np.int32)

            if confirmed_status == "occupied":
                color = (0, 0, 255)
            elif confirmed_status == "available":
                color = (0, 255, 0)
            else:
                color = (0, 255, 255)

            cv2.polylines(display, [pts], True, color, 3)

            label = (
                f"{space_name}: {confirmed_status} "
                f"bright={decision['brightness']:.0f} "
                f"{decision['reason']}"
            )

            x, y = roi_points[0]
            y = max(y - 10, 25)

            cv2.putText(
                display,
                label,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

        for box, class_name, confidence in detected_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                display,
                f"{class_name} {confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        if SHOW_WINDOW:
            cv2.imshow("Parking Detector", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        time.sleep(CHECK_INTERVAL_SECONDS)

    cap.release()

    if SHOW_WINDOW:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
EOF