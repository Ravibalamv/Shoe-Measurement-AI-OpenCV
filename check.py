import cv2

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)

    if cap.isOpened():
        ret, frame = cap.read()

        if ret:
            print(f"Camera found at index {i}")

            cv2.imshow(f"Camera {i}", frame)

            cv2.waitKey(0)

            cv2.destroyAllWindows()

        cap.release()