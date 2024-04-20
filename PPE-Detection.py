from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0) # For Webcam
cap.set(3, 1280)
cap.set(4, 720)

#cap = cv2.VideoCapture("../Videos/ppe-1-1.mp4")
model = YOLO("ppe.pt")
classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person',
              'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery', 'mini-van',
              'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']
myColor = (0, 0, 255)
paused = False

while True:
    if not paused:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1))

            #Confidence

            conf = math.ceil(box.conf[0] * 100) / 100

            #Class Name

            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if conf > 0.5:
                if currentClass == "NO-Hardhat" or currentClass == "NO-Safety Vest" or currentClass == "NO-Mask":
                    myColor = (0, 0, 255)

                elif currentClass == "Hardhat" or currentClass == "Safety Vest" or currentClass == "Mask":
                    myColor = (0, 255, 0)

                else:
                    myColor = (255, 0, 0)
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1.4, thickness=1,
                                   colorB=myColor, colorT=(255, 255, 255), colorR=myColor)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    cv2.imshow("Image", img)
    key = cv2.waitKey(0 if paused else 1)
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
    elif key == ord('n') and paused:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break

cap.release()
cv2.destroyAllWindows()
