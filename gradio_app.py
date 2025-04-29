import gradio as gr
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import threading
import time
import json
import pandas as pd
from typing import List, Tuple, Optional, Dict
import tempfile
from pathlib import Path
import pickle

# Initialize global variables
path = 'Images'
images = []
classNames = []
encodeListKnown = []
exit_flag = False
frame_skip = 2
frame_count = 0
ATTENDANCE_FILE = 'Attendance.csv'
ENCODINGS_FILE = 'face_encodings.pkl'

# Ensure attendance file exists with headers
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, 'w') as f:
        f.write("Name,Time,Date\n")

# Save encodings to file
def save_encodings(encodings, names):
    try:
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump((encodings, names), f)
        print("Encodings saved successfully")
    except Exception as e:
        print(f"Error saving encodings: {str(e)}")

# Load encodings from file
def load_encodings():
    try:
        if os.path.exists(ENCODINGS_FILE):
            with open(ENCODINGS_FILE, 'rb') as f:
                encodings, names = pickle.load(f)
            print("Encodings loaded successfully")
            return encodings, names
    except Exception as e:
        print(f"Error loading encodings: {str(e)}")
    return None, None

# Check if encodings need to be updated
def check_encodings_update():
    try:
        if not os.path.exists(ENCODINGS_FILE):
            return True
            
        # Get list of current image files
        current_images = set(os.listdir(path))
        
        # Load existing encodings
        encodings, names = load_encodings()
        if encodings is None:
            return True
            
        # Check if any images were added or removed
        saved_names = set(names)
        if current_images != saved_names:
            return True
            
        return False
    except Exception as e:
        print(f"Error checking encodings: {str(e)}")
        return True

# Load images
def load_images():
    global images, classNames
    print("Loading images...")
    myList = os.listdir(path)
    for cl in myList:
        try:
            img_path = f'{path}/{cl}'
            curImg = cv2.imread(img_path)
            
            if curImg is None:
                print(f"Warning: Could not read image {cl}")
                continue
                
            rgb_img = cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB)
            if rgb_img.dtype != np.uint8:
                rgb_img = np.array(rgb_img, dtype=np.uint8)
            
            images.append(rgb_img)
            classNames.append(os.path.splitext(cl)[0])
            print(f"Successfully loaded {cl}")
        except Exception as e:
            print(f"Error processing image {cl}: {str(e)}")

# Find encodings
def findEncodings(images):
    encodeList = []
    for i, img in enumerate(images):
        try:
            if img.dtype != np.uint8:
                img = np.array(img, dtype=np.uint8)
            
            face_locations = face_recognition.face_locations(img)
            if not face_locations:
                print(f"No faces found in image {i} ({classNames[i]})")
                continue
                
            encode = face_recognition.face_encodings(img, face_locations)
            if encode:
                encodeList.append(encode[0])
                print(f"Successfully encoded {classNames[i]}")
        except Exception as e:
            print(f"Error encoding image {i} ({classNames[i]}): {str(e)}")
    
    return encodeList

# Initialize the system with optimized encoding loading
def initialize_system():
    global encodeListKnown, classNames
    
    # Check if we need to update encodings
    if check_encodings_update():
        print("Generating new encodings...")
        load_images()
        encodeListKnown = findEncodings(images)
        save_encodings(encodeListKnown, classNames)
    else:
        print("Loading saved encodings...")
        encodeListKnown, classNames = load_encodings()
        if encodeListKnown is None:
            print("Failed to load encodings, generating new ones...")
            load_images()
            encodeListKnown = findEncodings(images)
            save_encodings(encodeListKnown, classNames)
    
    print(f'Encoding Complete. Found {len(encodeListKnown)} encodings.')

# Mark attendance
def markAttendance(name):
    try:
        with open(ATTENDANCE_FILE, 'a+') as f:
            f.seek(0)
            nameList = [line.split(',')[0] for line in f.readlines() if line.strip()]
            
            time_now = datetime.now()
            current_date = time_now.strftime("%d/%m/%Y")
            current_time = time_now.strftime("%H:%M:%S")
            
            name_date_exists = False
            for line in f.readlines():
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 3 and parts[0] == name and parts[2].strip() == current_date:
                        name_date_exists = True
                        break
            
            if name not in nameList or not name_date_exists:
                f.write(f'\n{name},{current_time},{current_date}')
                print(f"Marked attendance for {name}")
                return True
        return False
    except Exception as e:
        print(f"Error marking attendance: {str(e)}")
        return False

# Get attendance records
def get_attendance_records() -> pd.DataFrame:
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
        return df
    except Exception as e:
        print(f"Error reading attendance records: {str(e)}")
        return pd.DataFrame(columns=["Name", "Time", "Date"])

# Export attendance to CSV
def export_to_csv():
    try:
        df = get_attendance_records()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attendance_export_{timestamp}.csv"
        df.to_csv(filename, index=False)
        return f"Exported to {filename}"
    except Exception as e:
        return f"Error exporting to CSV: {str(e)}"

# Export attendance to JSON
def export_to_json():
    try:
        df = get_attendance_records()
        records = df.to_dict(orient='records')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attendance_export_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(records, f, indent=4)
        return f"Exported to {filename}"
    except Exception as e:
        return f"Error exporting to JSON: {str(e)}"

# Process image function
def process_image(img: np.ndarray) -> Tuple[np.ndarray, List[str], str]:
    global frame_count
    try:
        if img is None:
            return None, [], "No image provided"
            
        # Resize image for faster processing
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        
        if imgS.dtype != np.uint8:
            imgS = np.array(imgS, dtype=np.uint8)
            
        facesCurFrame = face_recognition.face_locations(imgS)
        recognized_names = []
        attendance_status = []
        
        if facesCurFrame:
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
            
            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                
                if len(faceDis) > 0:
                    matchIndex = np.argmin(faceDis)
                    conf_threshold = 0.6
                    
                    if matches[matchIndex] and faceDis[matchIndex] < conf_threshold:
                        name = classNames[matchIndex].upper()
                        recognized_names.append(name)
                        
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 250, 0), cv2.FILLED)
                        cv2.putText(img, f"{name} {1-faceDis[matchIndex]:.2f}", 
                                  (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 
                                  0.8, (255, 255, 255), 2)
                        
                        # Mark attendance
                        if markAttendance(name):
                            cv2.putText(img, "ATTENDANCE MARKED", 
                                      (x1 + 6, y2 + 30), cv2.FONT_HERSHEY_COMPLEX, 
                                      0.8, (0, 255, 0), 2)
                            attendance_status.append(f"{name}: Present")
                    else:
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(img, "UNKNOWN", 
                                  (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 
                                  0.8, (255, 255, 255), 2)
                        recognized_names.append("UNKNOWN")
        
        return img, recognized_names, "\n".join(attendance_status) if attendance_status else "No faces recognized"
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return img, [], f"Error processing image: {str(e)}"

# Initialize the system
initialize_system()

# Create Gradio interface with custom theme
with gr.Blocks(
    title="Face Recognition Attendance System",
    theme=gr.themes.Soft(
        primary_hue="gray",
        secondary_hue="gray",
        neutral_hue="gray",
        text_size="lg",
        spacing_size="lg",
        radius_size="lg",
    )
) as demo:
    # Custom CSS for additional styling
    gr.HTML("""
    <style>
        .gradio-container {
            background: #12263A ;
            color: #F4F4ED;
        }
        .gradio-container .panel {
            background: #F4F4ED;
            border-radius: 10px;
            padding: 20px;
            margin: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .gradio-container .button {
            background: #646E78 ;
            color: #F4F4ED;
            border: none;
            padding: 12px 24px;
            border-radius: 5px;
            transition: all 0.3s ease;
            font-weight: 500;
        }
        .gradio-container .button:hover {
            background: #7a858f;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .gradio-container .textbox {
            background: #F4F4ED;
            border: 2px solid #12263A ;
            border-radius: 5px;
            color: #12263A ;
        }
        .creator-credits {
            text-align: center;
            padding: 20px;
            background: rgba(244, 244, 237, 0.1);
            border-radius: 10px;
            margin-bottom: 20px;
            backdrop-filter: blur(5px);
        }
        .creator-credits h1 {
            color: #F4F4ED;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        .creator-credits p {
            color: #F4F4ED;
            font-size: 1.2em;
            opacity: 0.9;
        }
        .mobile-notice {
            display: none;
            text-align: center;
            padding: 10px;
            background: rgba(244, 244, 237, 0.1);
            border-radius: 5px;
            margin: 10px 0;
        }
        @media (max-width: 768px) {
            .mobile-notice {
                display: block;
            }
            .gradio-container .button {
                width: 100%;
                margin: 5px 0;
            }
            .gradio-container .panel {
                padding: 15px;
            }
        }
    </style>
    """)
    
    # Creator credits
    gr.HTML("""
    <div class="creator-credits">
        <h1>Face Recognition Attendance System</h1>
        <p>Created by Ayush and Soumay</p>
    </div>
    """)
    
    # Mobile notice
    gr.HTML("""
    <div class="mobile-notice">
        <p>For best experience, use in landscape mode on mobile devices</p>
    </div>
    """)
    
    gr.Markdown("Upload an image or use your webcam to detect faces and mark attendance.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="numpy")
            with gr.Row():
                submit_btn = gr.Button("Process Image", variant="primary")
                live_btn = gr.Button("Start Live Processing", variant="primary")
                stop_btn = gr.Button("Stop Live Processing", variant="secondary")
        
        with gr.Column():
            output_image = gr.Image(label="Processed Image", type="numpy")
            output_names = gr.Textbox(label="Recognized Names", lines=2)
            attendance_status = gr.Textbox(label="Attendance Status", lines=2)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Attendance Records")
            with gr.Row():
                view_records_btn = gr.Button("View Attendance Records", variant="primary")
                export_csv_btn = gr.Button("Export to CSV", variant="primary")
                export_json_btn = gr.Button("Export to JSON", variant="primary")
            records_display = gr.Dataframe(label="Attendance Records", headers=["Name", "Time", "Date"])
            export_status = gr.Textbox(label="Export Status", lines=2)
    
    def process_live():
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        
        while not exit_flag:
            success, img = cap.read()
            if not success:
                continue
                
            processed_img, names, status = process_image(img)
            yield processed_img, names, status
            time.sleep(0.1)
            
        cap.release()
    
    def stop_processing():
        global exit_flag
        exit_flag = True
        return None, None, "Processing stopped"
    
    def view_records():
        df = get_attendance_records()
        return df
    
    submit_btn.click(
        fn=process_image,
        inputs=[input_image],
        outputs=[output_image, output_names, attendance_status]
    )
    
    live_btn.click(
        fn=lambda: (setattr(globals(), 'exit_flag', False), None, None, None),
        outputs=[output_image, output_names, attendance_status]
    ).then(
        fn=process_live,
        outputs=[output_image, output_names, attendance_status]
    )
    
    stop_btn.click(
        fn=stop_processing,
        outputs=[output_image, output_names, attendance_status]
    )
    
    view_records_btn.click(
        fn=view_records,
        outputs=[records_display]
    )
    
    export_csv_btn.click(
        fn=export_to_csv,
        outputs=[export_status]
    )
    
    export_json_btn.click(
        fn=export_to_json,
        outputs=[export_status]
    )
    
    gr.Markdown("## API Usage")
    gr.Markdown("""
    This interface supports API access. You can make POST requests to the endpoint with an image file.
    
    Example using curl:
    ```bash
    curl -X POST -F "image=@path/to/image.jpg" http://localhost:7860/run/predict
    ```
    """)

if __name__ == "__main__":
    # Launch with share=True to get a public URL
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    ) 