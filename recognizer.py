import cv2

class FaceRecognizer(object):
	def __init__(self, face_recognizer='eigenface'):
		if face_recognizer == 'lbph':
			self.recognizer = cv2.face.createLBPHFaceRecognizer()
		elif face_recognizer == 'fisher':
			self.recognizer = cv2.face.createFisherFaceRecognizer()
		else:
			self.recognizer = cv2.face.createEigenFaceRecognizer() 

	def train(self, faces, labels):
		self.recognizer.train(faces, labels)

	def predict(self, face, labels_dic):
		collector = cv2.face.MinDistancePredictCollector()
		self.recognizer.predict(face, collector)
		return collector.getDist(), labels_dic[collector.getLabel()]