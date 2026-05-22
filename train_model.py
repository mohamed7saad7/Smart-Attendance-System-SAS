import os
import cv2
import numpy as np

from keras_facenet import FaceNet
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

from joblib import dump

# =====================================================
# FACENET
# =====================================================

embedder = FaceNet()

X = []

y = []

dataset_path = "Dataset"

# =====================================================
# READ DATASET
# =====================================================

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

        img = cv2.imread(
            image_path
        )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        img = cv2.resize(
            img,
            (160,160)
        )

        embedding = embedder.embeddings(
            [img]
        )[0]

        X.append(
            embedding
        )

        y.append(
            person_name
        )

# =====================================================
# NUMPY
# =====================================================

X = np.asarray(X)

y = np.asarray(y)

# =====================================================
# LABEL ENCODER
# =====================================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

# =====================================================
# SVM MODEL
# =====================================================

model = SVC(
    kernel='rbf',
    probability=True
)

# =====================================================
# TRAIN
# =====================================================

model.fit(
    X,
    y_encoded
)

# =====================================================
# SAVE MODELS
# =====================================================

os.makedirs(
    "models",
    exist_ok=True
)

dump(
    model,
    "models/svm_model.joblib"
)

dump(
    encoder,
    "models/label_encoder.joblib"
)

print(
    "Training Completed Successfully!"
)