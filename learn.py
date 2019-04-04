import cv2
import face_recognition
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
import cPickle

people = ['anderson', 'gadi', 'avihay', 'shlez', 'aviran']
#people = ['anderson', 'gadi', 'avihay', 'shlez']
#people = ['anderson', 'gadi', 'avihay']
#people = ['anderson', 'avihay']
#people = ['gadi', 'avihay']

X = []
y = []
for person in people:
	for img_file in os.listdir(person):
		try:
			img = cv2.cvtColor(cv2.imread('%s/%s' % (person, img_file)), cv2.COLOR_BGR2RGB)
		except:
			print('Error with %s/%s' % (person, img_file))
		boxes = face_recognition.face_locations(img, model='hog')
		if len(boxes) == 0:
			continue
		X.append(face_recognition.face_encodings(img, boxes)[0])
		y.append(person)

X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y)

#clf = LinearSVC()
#clf = SVC(probability=True, kernel='linear')
#clf = LogisticRegression()
#clf = KNeighborsClassifier(n_neighbors=30)
#clf = VotingClassifier(estimators=[('lr', LogisticRegression()), ('svc', SVC(probability=True, kernel='linear')), ('knn', KNeighborsClassifier(n_neighbors=30))], voting='soft')
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 50, 10))
clf = BaggingClassifier(KNeighborsClassifier(n_neighbors=15), max_samples=0.5, max_features=0.7)

clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)
print(accuracy_score(y_test, y_predict))


with open('model.pkl', 'wb') as f:
	cPickle.dump(clf, f)
