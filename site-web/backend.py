from flask import Flask, request, jsonify, render_template, Response
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load two YOLO models
try:
    model1 = YOLO("C:/Users/slimane/Downloads/best (8).pt")
    model2 = YOLO("C:/Users/slimane/Downloads/best (4).pt")  # Make sure this is the correct path
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    traceback.print_exc()

conf_threshold = 0.8


@app.route('/')
def index():
    return render_template('index.html')


def ensemble_detect(image, conf_threshold=0.8):
    """Run detection with both models and take highest confidence detections"""
    try:
        # Run inference with both models using the user-provided confidence threshold
        results1 = model1.predict(image, conf=conf_threshold)
        results2 = model2.predict(image, conf=conf_threshold)
        
        # Get detections from both models
        detections1 = []
        if len(results1) > 0 and results1[0].boxes is not None and len(results1[0].boxes) > 0:
            detections1 = results1[0].boxes.data.cpu().numpy()
        
        detections2 = []
        if len(results2) > 0 and results2[0].boxes is not None and len(results2[0].boxes) > 0:
            detections2 = results2[0].boxes.data.cpu().numpy()
        
        logger.debug(f"Model 1 detections: {len(detections1)}")
        logger.debug(f"Model 2 detections: {len(detections2)}")
        
        # Combine detections
        combined_detections = []
        
        # Process detections from first model
        for det in detections1:
            combined_detections.append(det)
        
        # Process detections from second model, looking for better confidence or new detections
        for det in detections2:
            x_min, y_min, x_max, y_max, conf, cls_id = det
            
            # Check if this is a similar detection to one already processed
            is_new_detection = True
            for i, existing_det in enumerate(combined_detections):
                ex_min, ey_min, ex_max, ey_max, ex_conf, ex_cls_id = existing_det
                
                # Calculate IoU to determine if it's the same object
                iou = calculate_iou(
                    [x_min, y_min, x_max, y_max],
                    [ex_min, ey_min, ex_max, ey_max]
                )
                
                # If same class and high overlap, keep the one with higher confidence
                if int(ex_cls_id) == int(cls_id) and iou > 0.5:
                    is_new_detection = False
                    if conf > ex_conf:
                        combined_detections[i] = det  # Replace with higher confidence detection
                    break
            
            # If this is a completely new detection, add it
            if is_new_detection:
                combined_detections.append(det)
        
        logger.debug(f"Combined detections: {len(combined_detections)}")
        return combined_detections
    except Exception as e:
        logger.error(f"Error in ensemble_detect: {str(e)}")
        traceback.print_exc()
        return []


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    try:
        x1, y1, x2, y2 = box1
        x1b, y1b, x2b, y2b = box2

        # Calculate intersection area
        x_left = max(x1, x1b)
        y_top = max(y1, y1b)
        x_right = min(x2, x2b)
        y_bottom = min(y2, y2b)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union area
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2b - x1b) * (y2b - y1b)
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU
        return intersection_area / union_area
    except Exception as e:
        logger.error(f"Error in calculate_iou: {str(e)}")
        traceback.print_exc()
        return 0.0


@app.route('/detect_video', methods=['POST'])
def detect_video():
    try:
        # Get confidence threshold from request (default to 0.8 if not provided)
        conf_threshold = float(request.form.get('confidence', 0.8))
        
        frame_file = request.files['frame']
        frame_data = np.frombuffer(frame_file.read(), np.uint8)
        frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Failed to decode frame'}), 400
            
        # Run detection with specified confidence threshold
        detections = ensemble_detect(frame, conf_threshold)

        # Draw detections on frame
        for det in detections:
            x_min, y_min, x_max, y_max, conf, cls_id = det
            
            # Get class name safely
            cls_id_int = int(cls_id)
            if cls_id_int in model1.names:
                label = model1.names[cls_id_int]
            elif cls_id_int in model2.names:
                label = model2.names[cls_id_int]
            else:
                label = f"Unknown-{cls_id_int}"
                
            display_text = f"{label} {conf:.2f}"
            
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.putText(frame, display_text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Format response with classes and confidence for frontend
        classes_with_confidence = []
        for det in detections:
            cls_id_int = int(det[5])
            if cls_id_int in model1.names:
                label = model1.names[cls_id_int]
            elif cls_id_int in model2.names:
                label = model2.names[cls_id_int]
            else:
                label = f"Unknown-{cls_id_int}"
                
            classes_with_confidence.append({
                "label": label,
                "confidence": float(det[4])
            })

        return jsonify({'image': image_base64, 'classes': classes_with_confidence})

    except Exception as e:
        error_msg = f"Error in detect_video: {str(e)}"
        logger.error(error_msg)
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500


@app.route('/detect_image', methods=['POST'])
def detect_image():
    try:
        # Get confidence threshold from request (default to 0.8 if not provided)
        conf_threshold = float(request.form.get('confidence', 0.8))
        
        logger.debug("Processing image upload")
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
            
        # Read image file
        image_data = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        logger.debug(f"Image shape: {image.shape}")
        
        # Run detection with specified confidence threshold
        detections = ensemble_detect(image, conf_threshold)
        logger.debug(f"Detections found: {len(detections)}")

        # Draw detections on image
        for det in detections:
            x_min, y_min, x_max, y_max, conf, cls_id = det
            
            # Get class name safely
            cls_id_int = int(cls_id)
            if cls_id_int in model1.names:
                label = model1.names[cls_id_int]
            elif cls_id_int in model2.names:
                label = model2.names[cls_id_int]
            else:
                label = f"Unknown-{cls_id_int}"
                
            display_text = f"{label} {conf:.2f}"
            
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.putText(image, display_text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Format response with classes and confidence for frontend
        classes_with_confidence = []
        for det in detections:
            cls_id_int = int(det[5])
            if cls_id_int in model1.names:
                label = model1.names[cls_id_int]
            elif cls_id_int in model2.names:
                label = model2.names[cls_id_int]
            else:
                label = f"Unknown-{cls_id_int}"
                
            classes_with_confidence.append({
                "label": label,
                "confidence": float(det[4])
            })
            
        logger.debug("Successfully processed image")
        return jsonify({'image': image_base64, 'classes': classes_with_confidence})

    except Exception as e:
        error_msg = f"Error in detect_image: {str(e)}"
        logger.error(error_msg)
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500


if __name__ == '__main__':
    app.run(debug=True)
