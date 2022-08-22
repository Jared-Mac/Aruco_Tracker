import cv2



cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

while(True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    if(cv2.waitKey(1) & 0xFF == ord('q')):
        cap.release()
        break
cap.release()
cv2.destroyAllWindows()
