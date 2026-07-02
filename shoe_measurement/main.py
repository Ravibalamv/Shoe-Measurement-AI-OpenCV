# main.py
import cv2
import numpy as np
from modules.camera import IndustrialCamera
from modules.detector import ShoeDetector
from modules.background import BackgroundRemover
from modules.contour import ContourFinder
from modules.measurement import ShoeMeasurement
from modules.tongue import TongueDetector

camera=IndustrialCamera(camera_index=1)
detector=ShoeDetector("best.pt")
background=BackgroundRemover()
contours=ContourFinder()
measurement=ShoeMeasurement()
tongue_detector=TongueDetector()

previous_tongue=None

while True:
    frame=camera.read()
    if frame is None:
        break

    results=detector.detect(frame)
    annotated=frame.copy()

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1,y1,x2,y2=map(int,box.xyxy[0])
            margin=20
            x1=max(0,x1-margin)
            y1=max(0,y1-margin)
            x2=min(frame.shape[1],x2+margin)
            y2=min(frame.shape[0],y2+margin)

            shoe_roi=frame[y1:y2,x1:x2]
            if shoe_roi.size==0:
                continue

            h,w=shoe_roi.shape[:2]
            dh=600
            dw=int(w*(dh/h))

            cv2.imshow("Cropped Shoe",cv2.resize(shoe_roi,(dw,dh)))

            mask=background.remove(shoe_roi)
            cv2.imshow("Binary Mask",cv2.resize(mask,(dw,dh)))

            contour=contours.find_largest(mask)
            if contour is None:
                continue

            contour_img=contours.draw(shoe_roi,contour)
            heel,toe,_,shoe_mm=measurement.measure(contour)
            measurement_img=measurement.draw(contour_img,heel,toe,shoe_mm)

            tongue_point,tongue_mask=tongue_detector.detect(shoe_roi)
            if tongue_point is not None:
                if previous_tongue is None:
                    previous_tongue=tongue_point
                alpha=0.8
                tongue_point=(
                    int(alpha*previous_tongue[0]+(1-alpha)*tongue_point[0]),
                    int(alpha*previous_tongue[1]+(1-alpha)*tongue_point[1])
                )
                previous_tongue=tongue_point

                measurement_img=tongue_detector.draw(measurement_img,tongue_point)

                # ----------------------------------
                # Tongue -> Blue Point Measurement
                # ----------------------------------
                blue_point = np.array(heel, dtype=np.float32)
                tongue = np.array(tongue_point, dtype=np.float32)

                pixel_distance = cv2.norm(blue_point, tongue)
                tongue_mm = pixel_distance * measurement.mm_per_pixel

                cv2.line(
                    measurement_img,
                    tuple(blue_point.astype(int)),
                    tuple(tongue.astype(int)),
                    (255,255,0),
                    3
                )

                cv2.putText(
                    measurement_img,
                    f"Tongue->Blue : {tongue_mm:.2f} mm",
                    (10,55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,0),
                    2
                )

                print("--------------------------------")
                print(f"Shoe Length     : {shoe_mm:.2f} mm")
                print(f"Tongue -> Blue  : {tongue_mm:.2f} mm")

                cv2.imshow("Tongue Mask",cv2.resize(tongue_mask,(dw,dh)))

            cv2.imshow("Measurement",cv2.resize(measurement_img,(dw,dh)))

            conf=float(box.conf[0])
            cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(annotated,f"Shoe {conf:.2f}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

    cv2.imshow("Industrial Camera",cv2.resize(annotated,(1280,720)))

    if cv2.waitKey(1)==27:
        break

camera.release()
cv2.destroyAllWindows()
