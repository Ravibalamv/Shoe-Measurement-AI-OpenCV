import cv2
import numpy as np
import glob
import os

# ==============================
# Checkerboard Settings
# ==============================

CHECKERBOARD = (9, 6)      # Inner corners
SQUARE_SIZE = 20.0         # mm

# ==============================
# Prepare object points
# ==============================

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

# ==============================
# Load images
# ==============================

images = glob.glob("calibration_images/*.jpg")

print(f"\nFound {len(images)} calibration images.\n")

if len(images) == 0:
    print("No images found!")
    exit()

os.makedirs("calibration_output", exist_ok=True)

success = 0

for fname in images:

    img = cv2.imread(fname)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        None
    )

    if ret:

        success += 1

        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11,11),
            (-1,-1),
            (
                cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )
        )

        imgpoints.append(corners2)

        cv2.drawChessboardCorners(
            img,
            CHECKERBOARD,
            corners2,
            ret
        )

        filename = os.path.join(
            "calibration_output",
            os.path.basename(fname)
        )

        cv2.imwrite(filename, img)

        print(f"Detected : {os.path.basename(fname)}")

    else:

        print(f"FAILED   : {os.path.basename(fname)}")

print("\n===================================")
print(f"Successful Images : {success}")
print("===================================\n")

# ==============================
# Camera Calibration
# ==============================

ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

print("Calibration RMS Error :", ret)

print("\nCamera Matrix:\n")
print(cameraMatrix)

print("\nDistortion Coefficients:\n")
print(dist)

# ==============================
# Save Calibration
# ==============================

np.savez(
    "calibration.npz",
    cameraMatrix=cameraMatrix,
    distCoeffs=dist,
    rvecs=rvecs,
    tvecs=tvecs
)

print("\nSaved calibration.npz")

# ==============================
# Undistort Sample Image
# ==============================

sample = cv2.imread(images[0])

h, w = sample.shape[:2]

newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
    cameraMatrix,
    dist,
    (w, h),
    1,
    (w, h)
)

undistorted = cv2.undistort(
    sample,
    cameraMatrix,
    dist,
    None,
    newCameraMatrix
)

cv2.imwrite(
    "calibration_output/undistorted_sample.jpg",
    undistorted
)

print("\nUndistorted sample saved.")
print("\nCalibration Complete!")