import os
import cv2
import numpy as np
import pandas as pd

from keras_facenet import FaceNet
from joblib import load
from datetime import datetime

from mtcnn import MTCNN


model = load("models/svm_model.joblib")


encoder = load("models/label_encoder.joblib")


embedder = FaceNet()


detector = MTCNN()


cap = cv2.VideoCapture(0)

attendance_marked = []

print("Camera Started...")

while True:

    # Frame
    ret, frame = cap.read()

    # RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    
    faces = detector.detect_faces(
        rgb_frame
    )

    for face_data in faces:

        x, y, w, h = face_data['box']

      
        face = frame[y:y+h, x:x+w]

        # Resize
        face = cv2.resize(
            face,
            (160, 160)
        )

        
        rgb_face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        
        embedding = embedder.embeddings(
            [rgb_face]
        )[0]

        
        embedding = np.expand_dims(
            embedding,
            axis=0
        )

        # Prediction
        prediction = model.predict(
            embedding
        )

        # Probability
        probability = np.max(
            model.predict_proba(embedding)
        )

       
        name = encoder.inverse_transform(
            prediction
        )[0]

        
        if probability < 0.90:

            name = "Unknown"

            color = (0, 0, 255)

        else:

            color = (0, 255, 0)

        
            if name not in attendance_marked:

                attendance_marked.append(name)

                print(f"{name} Attendance Marked")

        
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            color,
            2
        )

    
        cv2.putText(
            frame,
            name,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

    
    cv2.imshow(
        "Smart Attendance System",
        frame
    )

   
    if cv2.waitKey(1) == 27:
        break


cap.release()


cv2.destroyAllWindows()


attendance_data = []


now = datetime.now()


for student in attendance_marked:

    attendance_data.append({

        "Name": student,

        "Date": now.strftime("%Y-%m-%d"),

        "Time": now.strftime("%H:%M:%S")
    })


df = pd.DataFrame(attendance_data)


file_name = f"attendance/attendance_{now.strftime('%Y%m%d_%H%M%S')}.csv"

df.to_csv(
    file_name,
    index=False
)

print("Attendance Sheet Saved!")