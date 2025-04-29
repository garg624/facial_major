import gradio as gr
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import threading
import time
from typing import List, Tuple, Optional

# Initialize global variables
path = 'Images'
images = []
classNames = []
encodeListKnown = []
exit_flag = False
frame_skip = 2
frame_count = 0

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

# Mark attendance
def markAttendance(name):
    with open('Attendance.csv', 'a+') as f:
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

# Process image function
def process_image(img: np.ndarray) -> Tuple[np.ndarray, List[str], str]:
    global frame_count
    try:
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
        return img, [], "Error processing image"

# Initialize the system
load_images()
encodeListKnown = findEncodings(images)
print(f'Encoding Complete. Found {len(encodeListKnown)} encodings.')

# Create Gradio interface
with gr.Blocks(title="Face Recognition Attendance System") as demo:
    gr.Markdown("# Face Recognition Attendance System")
    gr.Markdown("Upload an image or use your webcam to detect faces and mark attendance.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="numpy")
            submit_btn = gr.Button("Process Image")
            live_btn = gr.Button("Start Live Processing")
            stop_btn = gr.Button("Stop Live Processing")
        
        with gr.Column():
            output_image = gr.Image(label="Processed Image", type="numpy")
            output_names = gr.Textbox(label="Recognized Names", lines=2)
            attendance_status = gr.Textbox(label="Attendance Status", lines=2)
    
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
    
    gr.Markdown("## API Usage")
    gr.Markdown("""
    This interface supports API access. You can make POST requests to the endpoint with an image file.
    
    Example using curl:
    ```bash
    curl -X POST -F "image=@path/to/image.jpg" http://localhost:7860/run/predict
    ```
    """)

if __name__ == "__main__":
    demo.launch(share=True) 