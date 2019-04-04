import cv2
import matplotlib.pyplot as plt
import face_recognition
import cPickle
import time
import os
import pyrebase

config = {
  "apiKey": "AIzaSyA_ckriMFaCyFsdSbboOYOgdWffAS-_vAk",
  "authDomain": "soluto-hack.firebaseapp.com",
  "databaseURL": "https://soluto-hack.firebaseio.com",
  "storageBucket": "soluto-hack.appspot.com",
  "serviceAccount": "soluto-hack-firebase-adminsdk-fbmsj-1d576d412c.json"
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()

MIN_TIME_SINCE_ANNOUNCED = 120
MIN_FRAMES = 3
LAST_X_SECONDS = 10

last_announced = {}
seen = {}

def announce(name):
	global last_announced
	t = time.time()
	if name in last_announced and t - last_announced[name] < MIN_TIME_SINCE_ANNOUNCED:
		return

	last_announced[name] = t
	os.system('say -v carmit "hi %s"' % name)
	db.child('kitchen').push({'name': name})



with open('model.pkl', 'rb') as f:
	clf = cPickle.load(f)

cam = cv2.VideoCapture(0)

BUFFER = 20

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


print('Ready!')
while True:
	val, img = cam.read()
	img = cv2.flip(img, 1)
	rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	t = time.time()
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
			if prob < 0.75:
				continue

			if name not in seen:
				seen[name] = []
			seen[name].append(t)
			print('Saw %s' % name)
	
	for name in seen:
		seen[name] = filter(lambda x: t - x <= LAST_X_SECONDS, seen[name])
		if len(seen[name]) >= MIN_FRAMES:
			announce(name)

