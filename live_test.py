# =====================================================
# LIVE MODEL TEST
# =====================================================

import cv2
import numpy as np

from keras_facenet import FaceNet
from joblib import load

# =====================================================
# LOAD MODEL
# =====================================================

model = load(
    "models/svm_model.joblib"
)

encoder = load(
    "models/label_encoder.joblib"
)

embedder = FaceNet()

# =====================================================
# FACE DETECTION
# =====================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# =====================================================
# CAMERA
# =====================================================

cap = cv2.VideoCapture(0)

print("Live Testing Started...")

# =====================================================
# LOOP
# =====================================================

while True:

    ret, frame = cap.read()

    frame = cv2.flip(
        frame,
        1
    )

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        # =============================================
        # FACE CROP
        # =============================================

        face = frame[
            y:y+h,
            x:x+w
        ]

        face = cv2.resize(
            face,
            (160,160)
        )

        rgb_face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        # =============================================
        # FACENET EMBEDDING
        # =============================================

        embedding = embedder.embeddings(
            [rgb_face]
        )[0]

        embedding = np.expand_dims(
            embedding,
            axis=0
        )

        # =============================================
        # PREDICTION
        # =============================================

        prediction = model.predict(
            embedding
        )

        probability = np.max(
            model.predict_proba(
                embedding
            )
        )

        confidence = probability * 100

        predicted_name = encoder.inverse_transform(
            prediction
        )[0]

        # =============================================
        # UNKNOWN DETECTION
        # =============================================

        if confidence < 70:

            predicted_name = "Unknown"

            color = (0,0,255)

        else:

            color = (0,255,0)

        # =============================================
        # RECTANGLE
        # =============================================

        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            color,
            2
        )

        # =============================================
        # SHOW NAME + CONFIDENCE
        # =============================================

        text = f"{predicted_name} ({confidence:.2f}%)"

        cv2.putText(
            frame,
            text,
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        # =============================================
        # PRINT TERMINAL
        # =============================================

        print(
            f"Prediction: {predicted_name} | Confidence: {confidence:.2f}%"
        )

    # =============================================
    # SHOW WINDOW
    # =============================================

    cv2.imshow(
        "LIVE MODEL TEST",
        frame
    )

    # ESC للخروج
    if cv2.waitKey(1) == 27:
        break

# =====================================================
# RELEASE
# =====================================================

cap.release()

cv2.destroyAllWindows()