import cv2
import json

CAMERA_INDEX = 0
OUTPUT_FILE = "parking_rois.json"

points = []
rois = []
current_space = 1

def mouse_callback(event, x, y, flags, param):
    global points, rois, current_space

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"Space {current_space} point added: {x}, {y}")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(points) >= 4:
            rois.append({
                "name": f"space_{current_space}",
                "points": points.copy()
            })
            print(f"Saved space_{current_space}: {points}")
            points.clear()
            current_space += 1
        else:
            print("Need at least 4 points before saving ROI.")

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("ROI Calibration")
cv2.setMouseCallback("ROI Calibration", mouse_callback)

print("Instructions:")
print("Left-click around parking space boundary.")
print("Right-click to save current parking space.")
print("Press 'q' after saving both spaces.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read camera.")
        break

    display = frame.copy()

    for roi in rois:
        pts = roi["points"]
        for p in pts:
            cv2.circle(display, tuple(p), 5, (0, 255, 0), -1)
        for i in range(len(pts)):
            cv2.line(display, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), (0, 255, 0), 2)
        cv2.putText(display, roi["name"], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    for p in points:
        cv2.circle(display, tuple(p), 5, (0, 0, 255), -1)

    for i in range(len(points) - 1):
        cv2.line(display, tuple(points[i]), tuple(points[i + 1]), (0, 0, 255), 2)

    cv2.imshow("ROI Calibration", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

with open(OUTPUT_FILE, "w") as f:
    json.dump(rois, f, indent=4)

print(f"Saved ROIs to {OUTPUT_FILE}")
