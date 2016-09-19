import cv2
import cv2.face
import utilities as ut
import video_camera
import detector

cv2.startWindowThread()
cv2.namedWindow("Tutorial", cv2.WINDOW_NORMAL)
webcam = video_camera.VideoCamera()
detector = detector.FaceDetector("classifiers/haarcascades/haarcascade_frontalface_default.xml")

while True:
	frame = webcam.get_frame()
	faces_coord = detector.detect(frame)
	
	for(x, y, w, h) in faces_coord:
		ut.draw_rectangle_with_text(frame, (x,y), (x+w,y+h))

	cv2.imshow("Tutorial", frame)

	# si se presiona ESC
	if cv2.waitKey(20) & 0xFF==27:
		break

cv2.destroyAllWindows()
