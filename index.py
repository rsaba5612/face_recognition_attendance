import face_recognition
import cmake
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture=cv2.VideoCapture(0)

#Load Known Faces

# Load Known Faces

saba_image = face_recognition.load_image_file("faces/saba.jpg")

encodings = face_recognition.face_encodings(saba_image)
print(f"Number of encodings found for maryam_image: {len(encodings)}")
if len(encodings) > 0:
    saba_encoding = encodings[0]
else:
    print("No face encodings were found for the given image. Please try another image.")

maryam_image = face_recognition.load_image_file("faces/maryam.jpg")

encodings = face_recognition.face_encodings(maryam_image)
print(f"Number of encodings found for maryam_image: {len(encodings)}")
if len(encodings) > 0:
    maryam_encoding = encodings[0]
else:
    print("No face encodings were found for the given image. Please try another image.")


known_face_encoding = [saba_encoding, maryam_encoding]

known_face_names = ["saba", "maryam"]


#list of expected students 
students= known_face_names.copy()

face_locations=[]
face_encoding=[]

#Get the current date and time

now=datetime.now()
current_date=now.strftime("%Y-%m-%d")

f=open(f"{current_date}.csv", "w+",newline="")
lnwriter =csv.writer(f)


while True:
 _, frame =video_capture.read()
small_frame=cv2.resize(frame,(0,0), fx=0.25,fy=0.25)
rgb_small_frame=cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

#Recognize faces
face_locations=face_recognition.face_locations(rgb_small_frame)
face_encoding=face_recognition.face_encodings(rgb_small_frame,face_locations)


for face_encoding in face_locations:
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    face_distance=face_recognition.face_distance(known_face_encoding,face_encodings)
best_match_index=np.argmin(face_distance)

if(matches[best_match_index]):
    name=known_face_names[best_match_index]


#Add the text if the person is present
if name in known_face_names:
   font=cv2.FONT_HERSHEY_SIMPLEX
   bottomLeftCornerOfText=(10,100)
   fontScale=1.5
   fontColor=(255,0,0)
   thickness=3
   lineType=2
   cv2.putText(frame,name + "Present",bottomLeftCornerOfText,font,fontScale,fontColor,thickness,lineType)

   if name in students:
      students.remove(name)
      current_time=now.strftime("%H-%M-%S")
      lnwriter.writerow([name,current_time])
while True:
    # capture frame-by-frame
    ret, frame = cap.read()

    # display the resulting frame
    cv2.imshow('frame',frame)

    # check for 'q' key pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
f.close