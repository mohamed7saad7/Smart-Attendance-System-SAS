# Smart-Attendance-System-SAS# 📷 Smart Attendance System (SAS)

Smart Attendance System (SAS) is an AI-based attendance management system designed to automate student attendance during lectures using Face Recognition and Computer Vision.

The system detects and recognizes registered students through a live camera feed, marks attendance automatically, and stores attendance records with date and time. This helps reduce manual work, save lecture time, and improve attendance accuracy.

---

## 📌 Project Overview

Traditional attendance methods such as manual roll-calls, paper sheets, or signatures can be slow, time-consuming, and sometimes inaccurate.

This project provides a smart solution by using face recognition technology to identify students in real time. Each student is registered in the dataset using their face images. After training the model, the system can recognize students during lectures and automatically record whether they are present or absent.

The system is suitable for schools, universities, training centers, and classroom environments.

---

## 🎯 Project Objectives

- Automate student attendance during lectures
- Reduce manual attendance recording
- Save time for instructors and students
- Improve accuracy in attendance tracking
- Detect present and absent students efficiently
- Provide a simple and practical GUI for using the system
- Store attendance records in an organized format

---

## 🛠️ Tech Stack & Tools

- **Programming Language:** Python
- **Computer Vision:** OpenCV
- **Face Embedding:** Keras FaceNet
- **Machine Learning:** Scikit-learn / SVM Classifier
- **Data Processing:** NumPy
- **Model Saving:** Joblib
- **GUI Framework:** Tkinter
- **Storage:** CSV / Attendance Files

---

## ⚙️ Core Features

### 1. Real-Time Face Detection
The system captures a live video stream from the camera and detects faces in real time.

### 2. Face Feature Extraction
Each detected face is processed and converted into facial embeddings using FaceNet. These embeddings represent the unique facial features of each student.

### 3. Student Recognition
The trained machine learning model compares the live face embeddings with the registered dataset and predicts the student's name.

### 4. Automatic Attendance Logging
When a student is recognized, the system automatically marks them as present and saves their attendance with the current date and time.

### 5. Duplicate Prevention
The system avoids recording the same student multiple times during the same attendance session.

### 6. User-Friendly Interface
The GUI allows the user to start the camera, run recognition, and manage attendance easily.

---

## 🧠 How the System Works

1. Student face images are collected and stored inside the dataset.
2. Each student has a separate folder named with their name or ID.
3. The training script extracts facial embeddings from the images.
4. The embeddings are used to train an SVM classification model.
5. The trained model is saved and used later for recognition.
6. During a lecture, the camera detects and recognizes students.
7. Recognized students are marked as present automatically.
8. Students who are not detected during the session can be considered absent.
9. Attendance records are saved for review and reporting.

---

## 📁 Dataset Structure

Each student must have a separate folder inside the `Dataset` folder.

```text
Dataset/
│
├── Student_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│
├── Student_2/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│
├── Student_3/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
