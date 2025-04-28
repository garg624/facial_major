import cv2
import numpy as np
import face_recognition
import os
import threading
from datetime import datetime

path = 'Images'
images = []
classNames = []

# Load images
print("Loading images...")
myList = os.listdir(path)
for cl in myList:
    try:
        # Read image
        img_path = f'{path}/{cl}'
        curImg = cv2.imread(img_path)
        
        if curImg is None:
            print(f"Warning: Could not read image {cl}")
            continue
            
        # Print debug information
        print(f"Image {cl}: Shape={curImg.shape}, Type={curImg.dtype}")
        
        # Convert to RGB and ensure 8-bit
        rgb_img = cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB)
        
        # Force conversion to uint8
        if rgb_img.dtype != np.uint8:
            rgb_img = np.array(rgb_img, dtype=np.uint8)
        
        # Check if the conversion was successful
        if rgb_img.dtype != np.uint8:
            print(f"Warning: Could not convert {cl} to uint8, skipping")
            continue
            
        # Store the RGB image and class name
        images.append(rgb_img)  # Store the RGB version
        classNames.append(os.path.splitext(cl)[0])
        print(f"Successfully loaded {cl}")
    except Exception as e:
        print(f"Error processing image {cl}: {str(e)}")

print(f"Trained images of students: {classNames}")

# Find encodings
def findEncodings(images):
    encodeList = []
    for i, img in enumerate(images):
        try:
            # img is already in RGB format and uint8 type from the loading step
            # Double-check to be absolutely sure
            if img.dtype != np.uint8:
                print(f"Warning: Image {i} is not uint8, converting...")
                img = np.array(img, dtype=np.uint8)
            
            # Get face encodings
            face_locations = face_recognition.face_locations(img)
            if not face_locations:
                print(f"No faces found in image {i} ({classNames[i]})")
                continue
                
            encode = face_recognition.face_encodings(img, face_locations)
            if encode:
                encodeList.append(encode[0])
                print(f"Successfully encoded {classNames[i]}")
            else:
                print(f"Could not encode face for {classNames[i]}")
        except Exception as e:
            print(f"Error encoding image {i} ({classNames[i]}): {str(e)}")
    
    return encodeList

# Mark attendance
def markAttendance(name):
    with open('Attendance.csv', 'a+') as f:
        f.seek(0)
        nameList = [line.split(',')[0] for line in f.readlines() if line.strip()]
        
        # Get current date and time
        time_now = datetime.now()
        current_date = time_now.strftime("%d/%m/%Y")
        current_time = time_now.strftime("%H:%M:%S")
        
        # Check if name is already in attendance for today
        name_date_exists = False
        for line in f.readlines():
            if line.strip():  # Skip empty lines
                parts = line.split(',')
                if len(parts) >= 3 and parts[0] == name and parts[2].strip() == current_date:
                    name_date_exists = True
                    break
        
        # Add to attendance if not already present today
        if name not in nameList or not name_date_exists:
            f.write(f'\n{name},{current_time},{current_date}')
            print(f"Marked attendance for {name}")

# Prepare encodings
print("Starting encoding process...")
encodeListKnown = findEncodings(images)
print(f'Encoding Complete. Found {len(encodeListKnown)} encodings.')

if len(encodeListKnown) == 0:
    print("No face encodings were generated. Please check your images.")
    exit()

# Setup webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height
frame_skip = 2  # Process every 2nd frame
frame_count = 0
exit_flag = False

# Process frames function
def process_frames():
    global frame_count, exit_flag
    while not exit_flag:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame from webcam")
            continue
            
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
            
        try:
            # Resize image for faster processing
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            
            # Convert to RGB (face_recognition requires RGB)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            
            # Ensure it's uint8 type
            if imgS.dtype != np.uint8:
                imgS = np.array(imgS, dtype=np.uint8)
                
            # Find faces in current frame
            facesCurFrame = face_recognition.face_locations(imgS)
            
            if facesCurFrame:
                # Get encodings for detected faces
                encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
                
                for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                    # Compare with known encodings
                    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                    
                    if len(faceDis) > 0:
                        matchIndex = np.argmin(faceDis)
                        
                        # Use a confidence threshold (lower distance = better match)
                        conf_threshold = 0.6
                        if matches[matchIndex] and faceDis[matchIndex] < conf_threshold:
                            name = classNames[matchIndex].upper()
                            
                            # Scale face location back to original size
                            y1, x2, y2, x1 = faceLoc
                            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                            
                            # Draw rectangle and name
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 250, 0), cv2.FILLED)
                            cv2.putText(img, f"{name} {1-faceDis[matchIndex]:.2f}", 
                                      (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 
                                      0.8, (255, 255, 255), 2)
                            
                            # Mark attendance
                            markAttendance(name)
                        else:
                            # Draw rectangle for unknown face
                            y1, x2, y2, x1 = faceLoc
                            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(img, "UNKNOWN", 
                                      (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 
                                      0.8, (255, 255, 255), 2)
            
            # Display the resulting frame
            cv2.imshow('Webcam', img)
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit_flag = True

# Run thread
print("Starting webcam...")
thread = threading.Thread(target=process_frames)
thread.start()

try:
    thread.join()
except KeyboardInterrupt:
    print("Keyboard interrupt detected")
    exit_flag = True
    thread.join()
finally:
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released")