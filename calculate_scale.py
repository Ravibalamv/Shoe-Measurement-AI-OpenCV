import cv2
import numpy as np
import json

# =====================================
# SETTINGS
# =====================================

image_path = "calibration_images/calib_007.jpg"   # <-- Change to your image path
pattern_size = (9, 6)                             # Inner corners
square_size_mm = 20.0                             # 20 mm squares

# =====================================
# LOAD IMAGE
# =====================================

img = cv2.imread(image_path)

if img is None:
    print("Image not found.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# =====================================
# FIND CHECKERBOARD
# =====================================

found, corners = cv2.findChessboardCorners(
    gray,
    pattern_size
)

if not found:
    print("Checkerboard not detected.")
    exit()

# Improve accuracy
criteria = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

corners = cv2.cornerSubPix(
    gray,
    corners,
    (11,11),
    (-1,-1),
    criteria
)

# =====================================
# CALCULATE PIXELS PER SQUARE
# =====================================

rows = pattern_size[1]
cols = pattern_size[0]

distances = []

# Horizontal distances
for r in range(rows):
    for c in range(cols - 1):

        p1 = corners[r*cols + c][0]
        p2 = corners[r*cols + c + 1][0]

        distances.append(np.linalg.norm(p1 - p2))

# Vertical distances
for r in range(rows - 1):
    for c in range(cols):

        p1 = corners[r*cols + c][0]
        p2 = corners[(r+1)*cols + c][0]

        distances.append(np.linalg.norm(p1 - p2))

avg_pixels = np.mean(distances)

mm_per_pixel = square_size_mm / avg_pixels
pixels_per_mm = avg_pixels / square_size_mm

print("\n============================")
print(f"Average pixels/square : {avg_pixels:.3f}")
print(f"mm per pixel          : {mm_per_pixel:.6f}")
print(f"pixels per mm         : {pixels_per_mm:.6f}")
print("============================")

# =====================================
# SAVE SCALE
# =====================================

data = {
    "mm_per_pixel": float(mm_per_pixel),
    "pixels_per_mm": float(pixels_per_mm)
}

with open("scale.json", "w") as f:
    json.dump(data, f, indent=4)

print("\nSaved scale.json")

# =====================================
# DISPLAY DETECTED CORNERS
# =====================================

cv2.drawChessboardCorners(
    img,
    pattern_size,
    corners,
    found
)

display = cv2.resize(img, (1280,720))

cv2.imshow("Detected Checkerboard", display)

cv2.waitKey(0)

cv2.destroyAllWindows()