import cv2

class FaceDetector(object):
	def __init__(self, xml_path):
		self.classifier = cv2.CascadeClassifier(xml_path)

	#min_neighbors cuantos vecinos deben ser falsos positivos para reconocer al objeto
	def detect(self, frame, biggest_only=True, scale_factor=1.2, min_neighbors=5, min_size=(30,30)):
		flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else \
				cv2.CASCADE_SCALE_IMAGE
		return self.classifier.detectMultiScale(frame, scaleFactor=scale_factor, minNeighbors=min_neighbors, \
			minSize=min_size, flags=flags)