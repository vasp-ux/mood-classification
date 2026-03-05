import cv2

url = "http://100.87.142.219:8080/video"  # CHANGE to your IP
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream not available")
        break
    
    cv2.imshow("Phone Camera", frame)

    if cv2.waitKey(1) == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
