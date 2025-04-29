import gradio as gr
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import tempfile
from typing import List, Tuple, Optional

# Initialize the model
class FaceRecognitionSystem:
    def __init__(self):
        self.path = 'Images'
        self.images = []
        self.classNames = []
        self.encodeListKnown = []
        self._load_images()
        self._train_model()

    def _load_images(self):
        print("Loading images...")
        myList = os.listdir(self.path)
        for cl in myList:
            try:
                img_path = f'{self.path}/{cl}'
                curImg = cv2.imread(img_path)
                
                if curImg is None:
                    print(f"Warning: Could not read image {cl}")
                    continue
                    
                rgb_img = cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB)
                if rgb_img.dtype != np.uint8:
                    rgb_img = np.array(rgb_img, dtype=np.uint8)
                
                self.images.append(rgb_img)
                self.classNames.append(os.path.splitext(cl)[0])
                print(f"Successfully loaded {cl}")
            except Exception as e:
                print(f"Error processing image {cl}: {str(e)}")

    def _train_model(self):
        print("Starting encoding process...")
        encodeList = []
        for i, img in enumerate(self.images):
            try:
                if img.dtype != np.uint8:
                    img = np.array(img, dtype=np.uint8)
                
                face_locations = face_recognition.face_locations(img)
                if not face_locations:
                    print(f"No faces found in image {i} ({self.classNames[i]})")
                    continue
                    
                encode = face_recognition.face_encodings(img, face_locations)
                if encode:
                    encodeList.append(encode[0])
                    print(f"Successfully encoded {self.classNames[i]}")
            except Exception as e:
                print(f"Error encoding image {i} ({self.classNames[i]}): {str(e)}")
        
        self.encodeListKnown = encodeList
        print(f'Encoding Complete. Found {len(self.encodeListKnown)} encodings.')

    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Process an image and return the annotated image and recognized names."""
        try:
            # Resize image for faster processing
            imgS = cv2.resize(image, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            
            if imgS.dtype != np.uint8:
                imgS = np.array(imgS, dtype=np.uint8)
                
            facesCurFrame = face_recognition.face_locations(imgS)
            recognized_names = []
            
            if facesCurFrame:
                encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
                
                for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                    matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace)
                    faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                    
                    if len(faceDis) > 0:
                        matchIndex = np.argmin(faceDis)
                        conf_threshold = 0.6
                        
                        if matches[matchIndex] and faceDis[matchIndex] < conf_threshold:
                            name = self.classNames[matchIndex].upper()
                            recognized_names.append(name)
                            
                            y1, x2, y2, x1 = faceLoc
                            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                            
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (0, 250, 0), cv2.FILLED)
                            cv2.putText(image, f"{name} {1-faceDis[matchIndex]:.2f}", 
                                      (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 
                                      0.8, (255, 255, 255), 2)
                            
                            # Mark attendance
                            self._mark_attendance(name)
                        else:
                            y1, x2, y2, x1 = faceLoc
                            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(image, "UNKNOWN", 
                                      (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 
                                      0.8, (255, 255, 255), 2)
                            recognized_names.append("UNKNOWN")
            
            return image, recognized_names
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return image, []

    def _mark_attendance(self, name: str):
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

# Initialize the face recognition system
face_system = FaceRecognitionSystem()

def process_image(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Process an image and return the annotated image and recognized names."""
    return face_system.process_image(image)

# Create Gradio interface
with gr.Blocks(title="Face Recognition System") as demo:
    gr.Markdown("# Face Recognition System")
    gr.Markdown("Upload an image or use your webcam to detect faces and mark attendance.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="numpy")
            submit_btn = gr.Button("Process Image")
        
        with gr.Column():
            output_image = gr.Image(label="Processed Image", type="numpy")
            output_names = gr.Textbox(label="Recognized Names", lines=2)
    
    submit_btn.click(
        fn=process_image,
        inputs=[input_image],
        outputs=[output_image, output_names]
    )
    
    gr.Markdown("## API Usage")
    gr.Markdown("""
    This interface also supports API access. You can make POST requests to the endpoint with an image file.
    
    Example using curl:
    ```bash
    curl -X POST -F "image=@path/to/image.jpg" http://localhost:7860/run/predict
    ```
    """)

if __name__ == "__main__":
    demo.launch(share=True) 