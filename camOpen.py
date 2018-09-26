import cv2
cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0
while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)
    if k%256 == 27:
         #Receiving escape to intimate closing webcam
        print("Escape entered .. Closing webcam..")
        break
    elif k%256 == 32:
        #Receiving space to intimate observe
        img_name = "observed_image{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} uploaded!".format(img_name))
        img_counter += 1
cam.release()
cv2.destroyAllWindows()
