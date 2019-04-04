import cv2
import matplotlib.pyplot as plt
import sys
from random import sample
import os

name = sys.argv[1]

cam = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

BUFFER = 20

images = []

while True:
	val, img = cam.read()
	img = cv2.flip(img, 1)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = list(faceCascade.detectMultiScale(img_gray, 1.3, 5))
	if not len(faces):
		continue
	faces.sort(key=lambda f: f[2] + f[3], reverse=True)
	x, y, w, h = faces[0]
	#for (x, y, w, h) in faces:
	#	cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
	face = img[y - BUFFER:y + h + BUFFER, x - BUFFER:x + w + BUFFER]
	
	

	#cv2.imshow('ook', img)
	try:
		cv2.imshow('ook', face)
		images.append(face)
	except:
		pass
	print(len(images))
	if cv2.waitKey(1) == 27:
		break

#chosen = sample(images, 100)
chosen = images
os.system('rm -rf %s' % name)
os.system('mkdir %s' % name)
for i, image in enumerate(chosen):
	cv2.imwrite('%s/%d.png' % (name, i), image)

cv2.destroyAllWindows()

