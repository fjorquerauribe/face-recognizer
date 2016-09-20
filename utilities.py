import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import video_camera
import detector

def plt_show(image, title=""):
	if len(image.shape) == 3:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	plt.axis("off")
	plt.title(title)
	plt.imshow(image, cmap="Greys_r")
	plt.show()

def draw_rectangle_with_text(frame, top_left, bottom_right, message=""):
	cv2.rectangle(frame, (top_left[0], top_left[1]), (bottom_right[0], bottom_right[1]), (150,150,0), 8)
	cv2.putText(frame, message, (top_left[0]-10,top_left[1]-10), cv2.FONT_HERSHEY_SIMPLEX, .7, (150,150,0), 2)

def cut_faces(frame, faces_coord):
	faces = []
	for (x, y, w, h) in faces_coord:
		w_rate = int(0.2 * w / 2)
		faces.append(frame[y: y+h, x+w_rate: x+w-w_rate])
	return faces

def normalize_intensity(faces):
	faces_norm = []
	for face in faces:
		if len(face.shape) == 3:
			face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
		faces_norm.append(cv2.equalizeHist(face))
	return faces_norm

def resize(faces, size=(50,50)):
	faces_norm = []
	for face in faces:
		if face.shape < size:
			face_norm = cv2.resize(face, size, interpolation = cv2.INTER_AREA)
		else:
			face_norm = cv2.resize(face, size, interpolation = cv2.INTER_CUBIC)
		faces_norm.append(face_norm)
	return faces_norm

def normalize_faces(frame, faces_coord):
	faces = cut_faces(frame, faces_coord)
	faces = normalize_intensity(faces)
	faces = resize(faces)
	return faces

def take_pictures():
	name = raw_input('Name: ').lower()
	files_path = 'pictures/' + name
	cv2.namedWindow('Save image', cv2.WINDOW_AUTOSIZE)
	webcam = video_camera.VideoCamera()
	mydetector = detector.FaceDetector('classifiers/haarcascades/haarcascade_frontalface_default.xml')
	if not os.path.exists(files_path):
		os.makedirs(files_path)
		counter = 0
		timer = 0
		while counter < 10:
			frame = webcam.get_frame()
			faces_coord = mydetector.detect(frame)
			if len(faces_coord) and timer % 700 == 50:
				faces = normalize_faces(frame, faces_coord)
				cv2.imwrite(files_path + '/' + str(counter) + '.jpg', faces[0])
				counter += 1
			draw_rectangle_with_text(frame, (faces_coord[0][0], faces_coord[0][1]), \
				(faces_coord[0][0]+faces_coord[0][2], faces_coord[0][1]+faces_coord[0][3]), name)
			cv2.imshow('Save image', frame)
			cv2.waitKey(50)
			timer += 50
		cv2.destroyAllWindows()
	else:
		print 'This name already has been used.'

def collect_faces():
	faces = []
	labels = []
	labels_dic = {}
	names = [name for name in os.listdir('pictures/') if not name.startswith('.')]
	for i, name in enumerate(names):
		labels_dic[i] = name
		for face in os.listdir('pictures/' + name):
			faces.append(cv2.imread('pictures/' + name + '/' + face, 0))
			labels.append(i)
	return (faces, np.array(labels), labels_dic)

