import cv2
import cv2.face
import utilities as ut
import video_camera
import detector
import recognizer
import numpy as np

cv2.startWindowThread()
cv2.namedWindow("Tutorial", cv2.WINDOW_NORMAL)
webcam = video_camera.VideoCamera()

#webcam.set(cv.CV_CAP_PROP_FRAME_WIDTH, int(800))
#webcam.set(cv.CV_CAP_PROP_FRAME_HEIGHT, int(600))

my_detector = detector.FaceDetector("classifiers/haarcascades/haarcascade_frontalface_default.xml")
my_recognizer = recognizer.FaceRecognizer('eigenface')

faces, labels, labels_dic = ut.collect_faces()

my_recognizer.train(faces, labels)
#ut.take_pictures()

while True:
	frame = webcam.get_frame()
	faces_coord = my_detector.detect(frame, min_neighbors=5)
	
	if len(faces_coord):
		faces = ut.normalize_faces(frame, faces_coord)

		for i, face in enumerate(faces):
			conf, pred = my_recognizer.predict(face, labels_dic)
			ut.draw_rectangle_with_text(frame, (faces_coord[i][0], faces_coord[i][1]),\
				( faces_coord[i][0]+faces_coord[i][2], faces_coord[i][1]+faces_coord[i][3] ), pred)
		cv2.imshow('Tutorial', frame)

	# si se presiona ESC
	if cv2.waitKey(20) & 0xFF==27:
		break

cv2.destroyAllWindows()
