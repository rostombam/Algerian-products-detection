
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from ultralytics import YOLO
import threading
import time
from PIL import Image, ImageTk
import numpy as np

class ObjectDetectionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLOv8 Object Detection")
        self.geometry("1000x600")
        self.configure(bg="#2C3E50")
        self.detection_active = False

        # Load YOLO models (replace with your model paths)
        self.model1 = YOLO('C:/Users/slimane/Downloads/best (8).pt')
        self.model2 = YOLO('C:/Users/slimane/Downloads/best (4).pt')
        self.names1 = self.model1.names
        self.names2 = self.model2.names
        self.num_classes1 = len(self.names1)
        self.combined_names = {**self.names1, **{k + self.num_classes1: v for k, v in self.names2.items()}}
        self.conf_threshold = 0.8
        self.nms_threshold = 0.5

        self.setup_style()
        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.exit_application)

    def setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Arial", 12, "bold"), padding=10, background="#3498DB", foreground="white")
        style.map("TButton", background=[("active", "#2980B9")])
        style.configure("Horizontal.TScale", background="#2C3E50")

    def create_widgets(self):
        # Top frame for title
        self.top_frame = tk.Frame(self, bg="#2C3E50")
        self.top_frame.pack(side="top", fill="x")
        tk.Label(self.top_frame, text="Object Detection", font=("Arial", 24, "bold"), fg="#ECF0F1", bg="#2C3E50").pack(pady=10)

        # Control frame for buttons and slider
        self.control_frame = tk.Frame(self, bg="#2C3E50")
        self.control_frame.pack(side="top", fill="x", padx=20, pady=10)
        tk.Label(self.control_frame, text="Confidence:", bg="#2C3E50", fg="#ECF0F1", font=("Arial", 12)).pack(side="left", padx=10)
        self.conf_scale = ttk.Scale(self.control_frame, from_=0.0, to=1.0, orient="horizontal", length=200)
        self.conf_scale.set(self.conf_threshold)
        self.conf_scale.pack(side="left", padx=10)
        self.conf_value_label = tk.Label(self.control_frame, text=f"{self.conf_threshold:.2f}", bg="#2C3E50", fg="#ECF0F1", font=("Arial", 12))
        self.conf_value_label.pack(side="left", padx=10)
        ttk.Button(self.control_frame, text="Import Image", command=self.detect_image).pack(side="left", padx=10)
        ttk.Button(self.control_frame, text="Start Real-time", command=self.start_real_time_detection).pack(side="left", padx=10)
        ttk.Button(self.control_frame, text="Exit", command=self.exit_application).pack(side="left", padx=10)

        # Main frame for center and right sections
        self.main_frame = tk.Frame(self, bg="#2C3E50")
        self.main_frame.pack(side="top", fill="both", expand=True)

        # Center frame for image display
        self.center_frame = tk.Frame(self.main_frame, bg="#2C3E50")
        self.center_frame.pack(side="left", fill="both", expand=True, padx=20, pady=20)
        self.result_label = tk.Label(self.center_frame, bg="#2C3E50")
        self.result_label.pack(expand=True, fill="both")

        # Right frame for detected classes
        self.right_frame = tk.Frame(self.main_frame, bg="#2C3E50", width=200)
        self.right_frame.pack(side="right", fill="y", padx=20, pady=20)
        tk.Label(self.right_frame, text="Detected Classes", font=("Arial", 14, "bold"), bg="#2C3E50", fg="#ECF0F1").pack(pady=10)
        self.classes_listbox = tk.Listbox(self.right_frame, font=("Arial", 12), bg="#34495E", fg="#ECF0F1", selectbackground="#2980B9", height=20)
        self.classes_listbox.pack(fill="both", expand=True)

        # Bottom frame for footer
        self.bottom_frame = tk.Frame(self, bg="#2C3E50")
        self.bottom_frame.pack(side="bottom", fill="x")
        tk.Label(self.bottom_frame, text="Powered by YOLOv8", font=("Arial", 10, "italic"), fg="#BDC3C7", bg="#2C3E50").pack(pady=5)

        # Bind confidence scale update
        self.conf_scale.config(command=self.update_conf)

    def update_conf(self, val):
        self.conf_threshold = float(val)
        self.conf_value_label.config(text=f"{self.conf_threshold:.2f}")

    def apply_nms(self, detections, nms_threshold):
        if len(detections) == 0:
            return detections
        idxs = np.argsort(detections[:, 4])[::-1]
        detections = detections[idxs]
        classes = np.unique(detections[:, 5])
        keep_indices = []
        for cls in classes:
            cls_indices = np.where(detections[:, 5] == cls)[0]
            cls_dets = detections[cls_indices]
            boxes = cls_dets[:, :4].astype(int).tolist()
            scores = cls_dets[:, 4].tolist()
            nms_indices = cv2.dnn.NMSBoxes(boxes, scores, 0, nms_threshold)
            if len(nms_indices) > 0:
                keep_indices.extend(cls_indices[nms_indices.flatten()])
        return detections[keep_indices]

    def draw_detections(self, frame, detections, scale_x, scale_y):
        for det in detections:
            x_min, y_min, x_max, y_max, conf, cls = det
            x_min, y_min, x_max, y_max = int(x_min * scale_x), int(y_min * scale_y), int(x_max * scale_x), int(y_max * scale_y)
            label = f"{self.combined_names[int(cls)]} {conf:.2f}"
            color = (0, 255, 0) if cls < self.num_classes1 else (255, 0, 0)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame

    def resize_image(self, image, max_width=700, max_height=400):
        h, w = image.shape[:2]
        scale = min(max_width / w, max_height / h)
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def cv2_to_tk(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.resize_image(image)
        img = Image.fromarray(image)
        return ImageTk.PhotoImage(img)

    def update_classes_list(self, detections):
        self.classes_listbox.delete(0, tk.END)
        if len(detections) > 0:
            classes = detections[:, 5]
            unique_classes = np.unique(classes)
            for cls in unique_classes:
                if cls < self.num_classes1:
                    model_name = "Model1"
                    class_name = self.names1[int(cls)]
                else:
                    model_name = "Model2"
                    class_name = self.names2[int(cls) - self.num_classes1]
                self.classes_listbox.insert(tk.END, f"{model_name}: {class_name}")

    def detect_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            image = cv2.imread(file_path)
            resized_image = cv2.resize(image, (640, 640))
            results1 = self.model1.predict(resized_image, conf=self.conf_threshold)
            results2 = self.model2.predict(resized_image, conf=self.conf_threshold)
            detections1 = results1[0].boxes.data.cpu().numpy() if results1[0].boxes is not None else np.empty((0, 6))
            detections2 = results2[0].boxes.data.cpu().numpy() if results2[0].boxes is not None else np.empty((0, 6))
            detections2[:, 5] += self.num_classes1
            combined_detections = np.vstack((detections1, detections2))
            combined_detections = self.apply_nms(combined_detections, self.nms_threshold)
            scale_x, scale_y = image.shape[1] / 640, image.shape[0] / 640
            annotated_image = self.draw_detections(image, combined_detections, scale_x, scale_y)
            self.update_classes_list(combined_detections)
            photo = self.cv2_to_tk(annotated_image)
            self.result_label.configure(image=photo)
            self.result_label.image = photo

    def real_time_detection(self):
        cap = cv2.VideoCapture(0)
        while self.detection_active:
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (640, 640))
            results1 = self.model1.predict(resized_frame, conf=self.conf_threshold)
            results2 = self.model2.predict(resized_frame, conf=self.conf_threshold)
            detections1 = results1[0].boxes.data.cpu().numpy() if results1[0].boxes is not None else np.empty((0, 6))
            detections2 = results2[0].boxes.data.cpu().numpy() if results2[0].boxes is not None else np.empty((0, 6))
            detections2[:, 5] += self.num_classes1
            combined_detections = np.vstack((detections1, detections2))
            combined_detections = self.apply_nms(combined_detections, self.nms_threshold)
            scale_x, scale_y = frame.shape[1] / 640, frame.shape[0] / 640
            annotated_frame = self.draw_detections(frame, combined_detections, scale_x, scale_y)
            self.update_classes_list(combined_detections)
            photo = self.cv2_to_tk(annotated_frame)
            self.result_label.configure(image=photo)
            self.result_label.image = photo
            self.update()
            time.sleep(0.01)
        cap.release()
        self.detection_active = False
        self.classes_listbox.delete(0, tk.END)

    def start_real_time_detection(self):
        if not self.detection_active:
            self.detection_active = True
            threading.Thread(target=self.real_time_detection, daemon=True).start()

    def exit_application(self):
        self.detection_active = False
        self.destroy()

if __name__ == "__main__":
    app = ObjectDetectionApp()
    app.mainloop()
