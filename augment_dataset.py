import os
import cv2



dataset_path = "Dataset"


for person_name in os.listdir(dataset_path):

    person_folder = os.path.join(
        dataset_path,
        person_name
    )

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(
            person_folder,
            image_name
        )

        img = cv2.imread(image_path)

        if img is None:
            continue


        brighter = cv2.convertScaleAbs(
            img,
            alpha=1.2,
            beta=30
        )

     
        darker = cv2.convertScaleAbs(
            img,
            alpha=0.8,
            beta=-20
        )

     
        blur = cv2.GaussianBlur(
            img,
            (5,5),
            0
        )

       
        flip = cv2.flip(
            img,
            1
        )

     
        base_name = image_name.split(".")[0]

        cv2.imwrite(
            f"{person_folder}/{base_name}_bright.jpg",
            brighter
        )

        cv2.imwrite(
            f"{person_folder}/{base_name}_dark.jpg",
            darker
        )

        cv2.imwrite(
            f"{person_folder}/{base_name}_blur.jpg",
            blur
        )

        cv2.imwrite(
            f"{person_folder}/{base_name}_flip.jpg",
            flip
        )

print("Dataset Augmentation Completed!")