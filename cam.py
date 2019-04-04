import cv2
import matplotlib.pyplot as plt
import face_recognition
import cPickle

with open('model.pkl', 'rb') as f:
	clf = cPickle.load(f)

cam = cv2.VideoCapture(0)

BUFFER = 20

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


while True:
	val, img = cam.read()
	img = cv2.flip(img, 1)
	rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(img_gray, 1.3, 5)
	for (x, y, w, h) in faces:
		cv2.rectangle(img, (x - BUFFER, y - BUFFER), (x + w + BUFFER, y + h + BUFFER), (0, 255, 0), 2)
		sub_img = img[y:y+h, x:x+w]
		boxes = face_recognition.face_locations(sub_img, model='hog')
		if len(boxes):
			encodings = face_recognition.face_encodings(sub_img, boxes)[0]
			X = [encodings]
			name = clf.predict(X)[0]
			prob = clf.predict_proba(X).max()
			if prob >= 0.75:
				cv2.putText(img, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

	cv2.imshow('ook', img)
	if cv2.waitKey(1) == 27:
		break

cv2.destroyAllWindows()

