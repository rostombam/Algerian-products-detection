Project Title: Algerian Products Detection
============================================

Description:
------------
This project implements an object detection system focused on detecting Algerian products using two YOLO models. The system features a Flask-based backend for serving detection endpoints and a modern, responsive web interface that supports both real-time detection (via webcam) and image upload detection. The ensemble method combines the detections of two models to improve accuracy.

Key Features:
-------------
- **Real-time Detection:** Capture and process live video feed from a webcam.
- **Image Upload:** Upload images for detection processing.
- **Ensemble YOLO Models:** Uses two YOLO models to combine detections and select the best predictions.
- **Confidence Threshold Adjustment:** Dynamic control over detection sensitivity.
- **Responsive Front-End:** A user-friendly interface built with HTML, Bootstrap, and JavaScript.

File Structure:
---------------
- **backend.py:** Contains the Flask server, YOLO model loading, endpoints (`/detect_video` and `/detect_image`), and detection logic including ensemble processing.
- **index.html:** Provides the web interface for both real-time detection and image uploads. Uses Bootstrap for styling and responsive design.
- **Product_Detection.ipynb:** A Jupyter Notebook that likely contains exploratory code or experiments related to product detection.
- **interface_tkinter.ipynb:** A Notebook for a Tkinter interface version, potentially offering a desktop-based alternative.

Setup and Installation:
-----------------------
1. **Prerequisites:**
   - Python 3.x installed on your system.
   - pip for managing Python packages.
   - (Optional) A virtual environment to isolate dependencies.

2. **Required Libraries:**
   - [Flask](https://flask.palletsprojects.com/) – for building the web server.
   - [OpenCV](https://opencv.org/) – for image and video processing.
   - [NumPy](https://numpy.org/) – for numerical operations.
   - [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) – for running YOLO-based object detection.
   - Additional libraries such as `base64` and `logging` (usually available in the standard Python library).

3. **Installation:**
   - Install the necessary packages using pip:
     ```
     pip install flask opencv-python numpy ultralytics
     ```
   - Ensure the YOLO model weights are available at the specified paths in **backend.py** (update the paths if necessary).

Usage:
------
1. **Starting the Server:**
   - Run the Flask application by executing:
     ```
     python backend.py
     ```
   - The server will start on `http://localhost:5000`.

2. **Accessing the Web Interface:**
   - Open a web browser and navigate to `http://localhost:5000` to access the detection interface.
   - Use the "Real-time Detection" tab to start webcam-based detection.
   - Use the "Image Upload" tab to select an image for object detection.

3. **Adjusting Settings:**
   - Use the provided sliders in the interface to adjust the confidence threshold, which filters detected objects based on prediction confidence.

Troubleshooting:
----------------
- If the camera feed is not working, ensure that your system has a compatible webcam and that you have granted the browser camera access.
- For any errors related to model loading or image processing, review the logs printed in the terminal to identify issues.

References:
-----------
- Flask Documentation: https://flask.palletsprojects.com/ :contentReference[oaicite:0]{index=0}
- OpenCV Official Site: https://opencv.org/ :contentReference[oaicite:1]{index=1}
- Ultralytics YOLO GitHub: https://github.com/ultralytics/ultralytics :contentReference[oaicite:2]{index=2}
- Bootstrap Documentation: https://getbootstrap.com/ :contentReference[oaicite:3]{index=3}

License:
--------
Specify your project's license here (e.g., MIT License) or add your own license terms.

Contact:
--------
Include contact information or instructions on how users can report issues or contribute to the project.

