import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    
    lower_light_blue = np.array([90, 50, 70])
    upper_light_blue = np.array([128, 255, 255])

    lower_orange = np.array([10, 100, 20])
    upper_orange = np.array([25, 255, 255])

    lower_red1 = np.array([0, 100, 20])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 20])
    upper_red2 = np.array([179, 255, 255])

    mask_light_blue = cv2.inRange(hsv, lower_light_blue, upper_light_blue)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    combined_mask = cv2.bitwise_or(mask_light_blue, mask_orange)
    combined_mask = cv2.bitwise_or(combined_mask, mask_red)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500: 
            x, y, w, h = cv2.boundingRect(contour)
            color_name = "Color"  

    
            if cv2.pointPolygonTest(contour, (x + w // 2, y + h // 2), False) >= 0:

                if mask_light_blue[y + h // 2, x + w // 2] > 0:
                    color_name = "Light Blue"
                elif mask_orange[y + h // 2, x + w // 2] > 0:
                    color_name = "Orange"
                elif mask_red[y + h // 2, x + w // 2] > 0:
                    color_name = "Red Maroon"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                color_name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    cv2.imshow("Original Frame with Boxes", frame)
    cv2.imshow("Filtered Colors Mask", combined_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    