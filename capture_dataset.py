import cv2
import os
from datetime import datetime

# ==========================================
# Camera Settings
# ==========================================
CAMERA_INDEX = 1

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

# Set desired resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Read actual resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Camera Resolution: {width} x {height}")

# ==========================================
# Create New Dataset Folder
# ==========================================

folder_name = "shoe_dataset_" + datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(folder_name, exist_ok=True)

print(f"\nImages will be saved in:\n{folder_name}\n")

# ==========================================
# Window
# ==========================================

cv2.namedWindow("Dataset Capture", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Dataset Capture", 1280, 720)

count = 0

print("===================================")
print("SPACE  -> Capture Image")
print("ESC    -> Exit")
print("===================================")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera.")
        break

    # Save original image
    original = frame.copy()

    # Resize only for display
    display = cv2.resize(frame, (1280, 720))

    h, w = display.shape[:2]

    # -----------------------------
    # Crosshair
    # -----------------------------
    cv2.line(display, (w//2, 0), (w//2, h), (0,255,0), 1)
    cv2.line(display, (0, h//2), (w, h//2), (0,255,0), 1)

    # -----------------------------
    # Text
    # -----------------------------
    cv2.putText(display,
                f"Captured: {count}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.putText(display,
                f"{width} x {height}",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,0),
                2)

    cv2.putText(display,
                "SPACE = Capture",
                (20,h-50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255),
                2)

    cv2.putText(display,
                "ESC = Exit",
                (20,h-15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255),
                2)

    cv2.imshow("Dataset Capture", display)

    key = cv2.waitKey(1) & 0xFF

    # SPACE
    if key == 32:

        filename = os.path.join(folder_name, f"shoe_{count:04d}.jpg")

        cv2.imwrite(filename, original)

        print(f"Saved: {filename}")

        count += 1

    # ESC
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("\n===================================")
print(f"Finished!")
print(f"Total Images Saved: {count}")
print(f"Saved in: {folder_name}")
print("===================================")