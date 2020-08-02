import streamlit as st 
import face_recognition
import dlib
import cv2
import time
import pickle 
from PIL import Image

st.markdown("Face recognition")

#load the known faces and embeddings
data = pickle.loads(open("encodings.pickle", "rb").read())

if st.checkbox("Start webcamera"):
	cap = cv2.VideoCapture(0)
	image_loc = st.empty()

	while cap.isOpened:
		ret, img = cap.read()
		time.sleep(0.01)
		#convert image from BGR (opencv ordering) to dlib ordering (RGB)
		rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


		#detect the (x,y) coordinate of the bounding boxes corresponding to each face in the input image
		boxes = face_recognition.face_locations(rgb_img,model="hog")
		# corresponding to each face in the input frame, then compute
		# the facial embeddings for each face
		encodings = face_recognition.face_encodings(rgb_img, boxes)
		names = []

		#loop over the facial embeddings
		for encoding in encodings:
			# attempt to match each face in the input image to our known encodings
			matches = face_recognition.compare_faces(data["encodings"], encoding)
			name = "Unknown"

			#check to see if we have found a match
			if True in matches:
				#find the indexes of all matched faces then initialize a 
				#dictionary to count the total number of times each face was matched
				matchedIdxs = [i for (i,b) in enumerate(matches) if b]
				counts = {}

				#loop over the matched indexes and maintain a count for each recognized face 
				for i in matchedIdxs:
					name = data["names"][i]
					counts[name] = counts.get(name, 0) + 1
				#determine the recognized face with the largest number of votes
				name = max(counts, key=counts.get)
			#update the list of names
			names.append(name)

		for (top, right, bottom, left) in boxes:
			top = int(top)
			right = int(right)
			bottom = int(bottom)
			left = int(left)

			cv2.rectangle(rgb_img, (left,top), (right,bottom), (0,255,0),2)
			y = top - 15 if top - 15 > 15 else top + 15
			cv2.putText(rgb_img, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
		#convert image to array and display
		img = Image.fromarray(rgb_img)
		image_loc.image(img)
		#if cv2.waitKey(1) & 0xFF == ord("q"):
		#	break

	cap.release()