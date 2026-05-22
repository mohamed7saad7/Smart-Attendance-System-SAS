import os
import cv2

from mtcnn import MTCNN


detector = MTCNN()


dataset_path = "Dataset"


for person_name in os.listdir(dataset_path):

    person_folder = os.path.join(
        dataset_path,
        person_name
    )

    print(f"Processing {person_name}...")

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(
            person_folder,
            image_name
        )

        img = cv2.imread(image_path)

        if img is None:
            continue

        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

    
        results = detector.detect_faces(
            rgb
        )

        if len(results) == 0:
            continue

        x, y, w, h = results[0]['box']

        x = max(0, x)
        y = max(0, y)

       
        face = img[
            y:y+h,
            x:x+w
        ]

       
        face = cv2.resize(
            face,
            (160,160)
        )

       
        cv2.imwrite(
            image_path,
            face
        )

print("Dataset Prepared Successfully!")