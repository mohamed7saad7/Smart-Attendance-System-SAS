# =========================================================
# SMART ATTENDANCE SYSTEM
# FINAL PROFESSIONAL VERSION
# =========================================================

import customtkinter as ctk
from tkinter import ttk

import cv2
import numpy as np
import pandas as pd
import os
import time

from collections import Counter
from PIL import Image

from keras_facenet import FaceNet
from joblib import load
from mtcnn import MTCNN

from datetime import datetime



ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")



class SmartAttendanceSystem(ctk.CTk):

    def __init__(self):

        super().__init__()

        # =================================================
        # WINDOW
        # =================================================

        self.title("Smart Attendance System")

        self.geometry("1700x950")

        self.configure(
            fg_color="#07111f"
        )

        # =================================================
        # VARIABLES
        # =================================================

        self.camera_running = False

        self.paused = False

        self.cap = None

        self.attendance_marked = []

        self.current_attendance_file = None

        # =================================================
        # TIMER FOR EACH PERSON
        # =================================================

        self.face_timers = {}

        # =================================================
        # BUFFER FOR EACH PERSON
        # =================================================

        self.person_buffers = {}

        # =================================================
        # GROUPS
        # =================================================

        self.groups = [
            "A",
            "B",
            "C",
            "D"
        ]

        # =================================================
        # SUBJECTS
        # =================================================

        self.subjects = [

            "Advanced Database",

            "Economtrics",

            "Machine Learning",

            "Managament of international Crisis",

            "Political Development",

            "Fesibility Study"
        ]

        # =================================================
        # LOAD MODELS
        # =================================================

        self.load_models()

        # =================================================
        # GRID
        # =================================================

        self.grid_columnconfigure(
            1,
            weight=1
        )

        self.grid_rowconfigure(
            0,
            weight=1
        )

        # =================================================
        # UI
        # =================================================

        self.create_sidebar()

        self.create_main_interface()

        self.create_analytics_panel()

        self.create_table()

        self.show_camera_view()

        self.update_analytics()

    # =====================================================
    # LOAD MODELS
    # =====================================================

    def load_models(self):

        self.model = load(
            "models/svm_model.joblib"
        )

        self.encoder = load(
            "models/label_encoder.joblib"
        )

        self.embedder = FaceNet()

        self.detector = MTCNN()

    # =====================================================
    # CREATE CSV
    # =====================================================
    



    # file_name = f"G:/My Drive/Attendance/attendance_{now.strftime('%H-%M-%S')}.csv"


    def create_attendance_file(self):

        selected_group = self.group_var.get()

        selected_subject = self.subject_var.get()

        subject_folder = os.path.join(
            "attendance",
            selected_group,
            selected_subject
        )

        os.makedirs(
            subject_folder,
            exist_ok=True
        )

        now = datetime.now()

        self.current_attendance_file = os.path.join(
            subject_folder,
            f"{now.strftime('%Y-%m-%d')}.csv"
        )

        df = pd.DataFrame(
            columns=["Name", "Time"]
        )

        df.to_csv(
            self.current_attendance_file,
            index=False
        )

    # =====================================================
    # SIDEBAR
    # =====================================================

    def create_sidebar(self):

        self.sidebar = ctk.CTkFrame(
            self,
            width=280,
            fg_color="#0b1727",
            corner_radius=25
        )

        self.sidebar.grid(
            row=0,
            column=0,
            padx=20,
            pady=20,
            sticky="ns"
        )

        # LOGO
        self.logo = ctk.CTkLabel(
            self.sidebar,
            text="🌀 SAS",
            font=ctk.CTkFont(
                size=40,
                weight="bold"
            ),
            text_color="#dbeafe"
        )

        self.logo.pack(
            pady=(30,20)
        )

        # CONTROL FRAME
        self.control_frame = ctk.CTkFrame(
            self.sidebar,
            fg_color="#111827",
            corner_radius=25
        )

        self.control_frame.pack(
            padx=15,
            pady=10,
            fill="x"
        )

        self.control_title = ctk.CTkLabel(
            self.control_frame,
            text="SAS Control Panel",
            font=ctk.CTkFont(
                size=24,
                weight="bold"
            ),
            text_color="#ffffff"
        )

        self.control_title.pack(
            pady=(25,25)
        )

        # GROUP
        self.group_label = ctk.CTkLabel(
            self.control_frame,
            text="Current Group",
            font=ctk.CTkFont(size=15),
            text_color="#9ca3af"
        )

        self.group_label.pack(
            anchor="w",
            padx=25
        )

        self.group_var = ctk.StringVar(
            value=self.groups[0]
        )

        self.group_menu = ctk.CTkOptionMenu(
            self.control_frame,
            values=self.groups,
            variable=self.group_var,
            height=45,
            corner_radius=14,
            fg_color="#1f2937",
            button_color="#3b82f6",
            button_hover_color="#2563eb",
            dropdown_fg_color="#111827"
        )

        self.group_menu.pack(
            padx=25,
            pady=(10,20),
            fill="x"
        )

        # SUBJECT
        self.subject_label = ctk.CTkLabel(
            self.control_frame,
            text="Current Course",
            font=ctk.CTkFont(size=15),
            text_color="#9ca3af"
        )

        self.subject_label.pack(
            anchor="w",
            padx=25
        )

        self.subject_var = ctk.StringVar(
            value=self.subjects[0]
        )

        self.subject_menu = ctk.CTkOptionMenu(
            self.control_frame,
            values=self.subjects,
            variable=self.subject_var,
            height=45,
            corner_radius=14,
            fg_color="#1f2937",
            button_color="#3b82f6",
            button_hover_color="#2563eb",
            dropdown_fg_color="#111827"
        )

        self.subject_menu.pack(
            padx=25,
            pady=(10,30),
            fill="x"
        )

        # START BUTTON
        self.start_btn = ctk.CTkButton(
            self.control_frame,
            text="▶ START ATTENDANCE",
            height=60,
            corner_radius=30,
            fg_color="#1e293b",
            border_width=3,
            border_color="#4ade80",
            text_color="#bbf7d0",
            font=ctk.CTkFont(
                size=17,
                weight="bold"
            ),
            command=self.start_attendance
        )

        self.start_btn.pack(
            padx=25,
            pady=(0,20),
            fill="x"
        )

        # PAUSE BUTTON
        self.pause_btn = ctk.CTkButton(
            self.control_frame,
            text="⏸ PAUSE SYSTEM",
            height=55,
            corner_radius=20,
            fg_color="#f59e0b",
            hover_color="#d97706",
            font=ctk.CTkFont(
                size=16,
                weight="bold"
            ),
            command=self.pause_attendance
        )

        self.pause_btn.pack(
            padx=25,
            pady=(0,20),
            fill="x"
        )

        # STOP BUTTON
        self.stop_btn = ctk.CTkButton(
            self.control_frame,
            text="■ STOP ATTENDANCE",
            height=55,
            corner_radius=20,
            fg_color="#ef4444",
            hover_color="#dc2626",
            font=ctk.CTkFont(
                size=16,
                weight="bold"
            ),
            command=self.stop_attendance
        )

        self.stop_btn.pack(
            padx=25,
            pady=(0,20),
            fill="x"
        )

        # STATUS
        self.status_label = ctk.CTkLabel(
            self.control_frame,
            text="SYSTEM READY",
            font=ctk.CTkFont(
                size=18,
                weight="bold"
            ),
            text_color="#dbeafe"
        )

        self.status_label.pack(
            pady=(10,25)
        )

        # EXIT
        self.exit_btn = ctk.CTkButton(
            self.sidebar,
            text="Exit",
            height=55,
            corner_radius=20,
            fg_color="#6b7280",
            hover_color="#4b5563",
            font=ctk.CTkFont(
                size=18,
                weight="bold"
            ),
            command=self.exit_system
        )

        self.exit_btn.pack(
            side="bottom",
            padx=25,
            pady=30,
            fill="x"
        )

    # =====================================================
    # MAIN INTERFACE
    # =====================================================

    def create_main_interface(self):

        self.main_frame = ctk.CTkFrame(
            self,
            fg_color="transparent"
        )

        self.main_frame.grid(
            row=0,
            column=1,
            sticky="nsew"
        )

        self.main_frame.grid_columnconfigure(
            0,
            weight=3
        )

        self.main_frame.grid_columnconfigure(
            1,
            weight=1
        )

        self.main_frame.grid_rowconfigure(
            1,
            weight=1
        )

        # TITLE
        self.header = ctk.CTkLabel(
            self.main_frame,
            text="SMART ATTENDANCE SYSTEM",
            font=ctk.CTkFont(
                size=42,
                weight="bold"
            ),
            text_color="#dbeafe"
        )

        self.header.grid(
            row=0,
            column=0,
            columnspan=2,
            pady=(25,25)
        )

        # CAMERA FRAME
        self.camera_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color="#111827",
            corner_radius=30,
            border_width=2,
            border_color="#3b82f6"
        )

        self.camera_frame.grid(
            row=1,
            column=0,
            padx=20,
            pady=20,
            sticky="nsew"
        )

        self.camera_label = ctk.CTkLabel(
            self.camera_frame,
            text=""
        )

        self.camera_label.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=20
        )

    # =====================================================
    # ANALYTICS PANEL
    # =====================================================

    def create_analytics_panel(self):

        self.analytics_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color="#0f172a",
            corner_radius=25
        )

        self.analytics_frame.grid(
            row=1,
            column=1,
            padx=(0,20),
            pady=20,
            sticky="nsew"
        )

        self.analytics_title = ctk.CTkLabel(
            self.analytics_frame,
            text="Analytics Dashboard",
            font=ctk.CTkFont(
                size=24,
                weight="bold"
            )
        )

        self.analytics_title.pack(
            pady=(25,25)
        )

        # PRESENT
        self.present_number = ctk.CTkLabel(
            self.analytics_frame,
            text="0",
            font=ctk.CTkFont(
                size=40,
                weight="bold"
            ),
            text_color="#4ade80"
        )

        self.present_number.pack(
            pady=(10,10)
        )

        self.present_text = ctk.CTkLabel(
            self.analytics_frame,
            text="Present Students",
            font=ctk.CTkFont(size=16)
        )

        self.present_text.pack()

        # ABSENT
        self.absent_number = ctk.CTkLabel(
            self.analytics_frame,
            text="0",
            font=ctk.CTkFont(
                size=40,
                weight="bold"
            ),
            text_color="#f87171"
        )

        self.absent_number.pack(
            pady=(30,10)
        )

        self.absent_text = ctk.CTkLabel(
            self.analytics_frame,
            text="Absent Students",
            font=ctk.CTkFont(size=16)
        )

        self.absent_text.pack()

        # LAST RECOGNIZED STUDENTS
        self.last_frame = ctk.CTkFrame(
            self.analytics_frame,
            fg_color="#1e293b",
            corner_radius=20
        )

        self.last_frame.pack(
            padx=20,
            pady=20,
            fill="both",
            expand=True
        )

        self.last_title = ctk.CTkLabel(
            self.last_frame,
            text="Last Recognized Students",
            font=ctk.CTkFont(
                size=18,
                weight="bold"
            ),
            text_color="#ffffff"
        )

        self.last_title.pack(
            pady=(15,15)
        )

        self.last_students_box = ctk.CTkTextbox(
            self.last_frame,
            fg_color="#0f172a",
            corner_radius=15,
            font=("Segoe UI",14),
            text_color="#dbeafe"
        )

        self.last_students_box.pack(
            padx=15,
            pady=(0,15),
            fill="both",
            expand=True
        )

    # =====================================================
    # TABLE
    # =====================================================

    def create_table(self):

        self.table_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color="#111827",
            corner_radius=30
        )

        style = ttk.Style()

        style.theme_use("default")

        style.configure(
            "Treeview",
            background="#1e293b",
            foreground="white",
            rowheight=45,
            fieldbackground="#1e293b",
            font=("Segoe UI",12)
        )

        self.table = ttk.Treeview(
            self.table_frame,
            columns=("Name","Time"),
            show="headings"
        )

        self.table.heading(
            "Name",
            text="Student Name"
        )

        self.table.heading(
            "Time",
            text="Attendance Time"
        )

        self.table.pack(
            fill="both",
            expand=True,
            padx=30,
            pady=30
        )

    # =====================================================
    # UPDATE ANALYTICS
    # =====================================================

    def update_analytics(self):

        total_students = len(
            os.listdir("dataset")
        )

        present_students = len(
            self.attendance_marked
        )

        absent_students = total_students - present_students

        self.present_number.configure(
            text=str(present_students)
        )

        self.absent_number.configure(
            text=str(absent_students)
        )

    # =====================================================
    # SHOW CAMERA
    # =====================================================

    def show_camera_view(self):

        self.table_frame.grid_forget()

        self.camera_frame.grid(
            row=1,
            column=0,
            padx=20,
            pady=20,
            sticky="nsew"
        )

    # =====================================================
    # SHOW TABLE
    # =====================================================

    def show_table_view(self):

        self.camera_frame.grid_forget()

        self.table_frame.grid(
            row=1,
            column=0,
            padx=20,
            pady=20,
            sticky="nsew"
        )

    # =====================================================
    # START
    # =====================================================

    def start_attendance(self):

        if self.camera_running:
            return

        self.camera_running = True

        self.paused = False

        self.group_menu.configure(
            state="disabled"
        )

        self.subject_menu.configure(
            state="disabled"
        )

        if self.current_attendance_file is None:

            self.attendance_marked.clear()

            self.face_timers.clear()

            self.person_buffers.clear()

            self.last_students_box.delete(
                "1.0",
                "end"
            )

            for item in self.table.get_children():

                self.table.delete(item)

            self.create_attendance_file()

        self.cap = cv2.VideoCapture(
            0,
            cv2.CAP_MSMF
        )

        self.status_label.configure(
            text="SYSTEM RUNNING"
        )

        self.show_camera_view()

        self.update_camera()

    # =====================================================
    # PAUSE
    # =====================================================

    def pause_attendance(self):

        if not self.camera_running:
            return

        self.paused = not self.paused

        if self.paused:

            self.pause_btn.configure(
                text="▶ RESUME SYSTEM",
                fg_color="#22c55e"
            )

            self.status_label.configure(
                text="SYSTEM PAUSED"
            )

        else:

            self.pause_btn.configure(
                text="⏸ PAUSE SYSTEM",
                fg_color="#f59e0b"
            )

            self.status_label.configure(
                text="SYSTEM RUNNING"
            )

            self.update_camera()

    # =====================================================
    # STOP
    # =====================================================

    def stop_attendance(self):

        self.camera_running = False

        self.paused = False

        if self.cap is not None:

            self.cap.release()

        self.current_attendance_file = None

        self.group_menu.configure(
            state="normal"
        )

        self.subject_menu.configure(
            state="normal"
        )

        self.pause_btn.configure(
            text="⏸ PAUSE SYSTEM",
            fg_color="#f59e0b"
        )

        self.status_label.configure(
            text="SYSTEM STOPPED"
        )

        self.show_table_view()

    # =====================================================
    # UPDATE CAMERA
    # =====================================================

    def update_camera(self):

        if not self.camera_running:
            return

        if self.paused:
            return

        ret, frame = self.cap.read()

        if ret:

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = self.detector.detect_faces(
                rgb_frame
            )

            for result in results:

                x, y, w, h = result['box']

                x = max(0, x)
                y = max(0, y)

                face = frame[y:y+h, x:x+w]

                try:

                    face = cv2.resize(
                        face,
                        (160,160)
                    )

                    rgb_face = cv2.cvtColor(
                        face,
                        cv2.COLOR_BGR2RGB
                    )

                    embedding = self.embedder.embeddings(
                        [rgb_face]
                    )[0]

                    embedding = np.expand_dims(
                        embedding,
                        axis=0
                    )

                    prediction = self.model.predict(
                        embedding
                    )

                    probability = np.max(
                        self.model.predict_proba(
                            embedding
                        )
                    )

                    confidence = probability * 100

                    predicted_name = self.encoder.inverse_transform(
                        prediction
                    )[0]

                    if confidence < 90:

                        predicted_name = "Unknown"

                        color = (0,0,255)

                    else:

                        color = (0,255,0)

                        # BUFFER FOR EACH PERSON
                        if predicted_name not in self.person_buffers:

                            self.person_buffers[predicted_name] = []

                        self.person_buffers[predicted_name].append(
                            predicted_name
                        )

                        if len(self.person_buffers[predicted_name]) > 10:

                            self.person_buffers[predicted_name].pop(0)

                        most_common_name = Counter(

                            self.person_buffers[predicted_name]

                        ).most_common(1)[0][0]

                        # TIMER FOR EACH PERSON
                        current_time_seconds = time.time()

                        if most_common_name not in self.face_timers:

                            self.face_timers[most_common_name] = current_time_seconds

                        elapsed_time = (

                            current_time_seconds -

                            self.face_timers[most_common_name]
                        )

                        # VERIFIED
                        if elapsed_time >= 1:

                            if most_common_name not in self.attendance_marked:

                                self.attendance_marked.append(
                                    most_common_name
                                )

                                attendance_time = datetime.now().strftime(
                                    "%H:%M:%S"
                                )

                                # TABLE
                                self.table.insert(
                                    "",
                                    "end",
                                    values=(
                                        most_common_name,
                                        attendance_time
                                    )
                                )

                                # LAST STUDENTS
                                self.last_students_box.insert(
                                    "end",
                                    f"{most_common_name} \n {attendance_time}\n"
                                )

                                # CSV
                                new_data = pd.DataFrame([{
                                    "Name": most_common_name,
                                    "Time": attendance_time
                                }])

                                new_data.to_csv(
                                    self.current_attendance_file,
                                    mode='a',
                                    header=False,
                                    index=False
                                )

                                self.update_analytics()

                    # RECTANGLE
                    cv2.rectangle(
                        frame,
                        (x,y),
                        (x+w,y+h),
                        color,
                        2
                    )

                    # TEXT
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

                except:
                    pass

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            img = Image.fromarray(rgb)

            ctk_image = ctk.CTkImage(
                light_image=img,
                dark_image=img,
                size=(1000,650)
            )

            self.camera_label.configure(
                image=ctk_image
            )

        self.after(
            10,
            self.update_camera
        )

    # =====================================================
    # EXIT
    # =====================================================

    def exit_system(self):

        self.camera_running = False

        if self.cap is not None:

            self.cap.release()

        self.destroy()


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app = SmartAttendanceSystem()

    app.mainloop()