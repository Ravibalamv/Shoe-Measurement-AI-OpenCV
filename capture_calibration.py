import cv2
import os

# ============================
# Camera Settings
# ============================
CAMERA_INDEX = 1

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

# Set desired resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Read actual resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Camera Resolution : {width} x {height}")

# ============================
# Create folder
# ============================

save_folder = "calibration_images"

os.makedirs(save_folder, exist_ok=True)

count = 0

# ============================
# Create Resizable Window
# ============================

cv2.namedWindow("Calibration Capture", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Calibration Capture", 1280, 720)

print("\n===================================")
print("SPACE  -> Capture Image")
print("ESC    -> Exit")
print("===================================\n")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera not detected!")
        break

    # Keep original frame
    original = frame.copy()

    # Resize ONLY for display
    display = cv2.resize(frame, (1280, 720))

    h, w = display.shape[:2]

    # --------------------------------
    # Crosshair
    # --------------------------------

    cv2.line(display,
             (w//2, 0),
             (w//2, h),
             (0,255,0),
             1)

    cv2.line(display,
             (0,h//2),
             (w,h//2),
             (0,255,0),
             1)

    # --------------------------------
    # Text
    # --------------------------------

    cv2.putText(display,
                f"Captured : {count}",
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

    # --------------------------------
    # Show
    # --------------------------------

    cv2.imshow("Calibration Capture", display)

    key = cv2.waitKey(1) & 0xFF

    # SPACE
    if key == 32:

        filename = os.path.join(
            save_folder,
            f"calib_{count:03d}.jpg"
        )

        # Save ORIGINAL FULL RESOLUTION image
        cv2.imwrite(filename, original)

        print(f"Saved -> {filename}")

        count += 1

    # ESC
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("\nFinished.")
print(f"Total Images Saved : {count}")