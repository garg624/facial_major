import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

def load_reference_images(path):
    """Load and encode reference images from the specified path."""
    images = []
    classNames = []
    
    print("Loading reference images...")
    myList = os.listdir(path)
    for cl in myList:
        try:
            img_path = f'{path}/{cl}'
            curImg = cv2.imread(img_path)
            
            if curImg is None:
                print(f"Warning: Could not read image {cl}")
                continue
                
            # Convert to RGB and ensure 8-bit
            rgb_img = cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB)
            if rgb_img.dtype != np.uint8:
                rgb_img = np.array(rgb_img, dtype=np.uint8)
            
            images.append(rgb_img)
            classNames.append(os.path.splitext(cl)[0])
            print(f"Successfully loaded {cl}")
        except Exception as e:
            print(f"Error processing image {cl}: {str(e)}")
    
    return images, classNames

def find_encodings(images):
    """Generate face encodings for the reference images."""
    encodeList = []
    for i, img in enumerate(images):
        try:
            if img.dtype != np.uint8:
                img = np.array(img, dtype=np.uint8)
            
            face_locations = face_recognition.face_locations(img)
            if not face_locations:
                print(f"No faces found in image {i}")
                continue
                
            encode = face_recognition.face_encodings(img, face_locations)
            if encode:
                encodeList.append(encode[0])
                print(f"Successfully encoded image {i}")
        except Exception as e:
            print(f"Error encoding image {i}: {str(e)}")
    
    return encodeList

def mark_attendance(name, output_file):
    """Mark attendance in the CSV file."""
    with open(output_file, 'a+') as f:
        f.seek(0)
        nameList = [line.split(',')[0] for line in f.readlines() if line.strip()]
        
        # Get current date and time
        time_now = datetime.now()
        current_date = time_now.strftime("%d/%m/%Y")
        current_time = time_now.strftime("%H:%M:%S")
        
        # Check if name is already in attendance for today
        name_date_exists = False
        f.seek(0)
        for line in f.readlines():
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 3 and parts[0] == name and parts[2].strip() == current_date:
                    name_date_exists = True
                    break
        
        # Add to attendance if not already present today
        if name not in nameList or not name_date_exists:
            f.write(f'\n{name},{current_time},{current_date}')
            print(f"Marked attendance for {name}")
            return True
    return False

def process_input_images(input_folder, reference_images_path, output_file):
    """Process images from input folder and mark attendance."""
    # Load reference images and their encodings
    reference_images, classNames = load_reference_images(reference_images_path)
    encodeListKnown = find_encodings(reference_images)
    
    if len(encodeListKnown) == 0:
        print("No face encodings were generated. Please check your reference images.")
        return
    
    # Create output file with header if it doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            f.write('Name,Time,Date')
    
    # Process each image in the input folder
    input_images = os.listdir(input_folder)
    total_images = len(input_images)
    total_faces_processed = 0
    total_attendance_marked = 0
    
    print(f"\nStarting to process {total_images} images from {input_folder}...")
    
    for img_name in input_images:
        try:
            img_path = os.path.join(input_folder, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Could not read image: {img_name}")
                continue
            
            # Convert to RGB and resize for faster processing
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            
            # Find faces in the image
            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
            
            print(f"\nProcessing {img_name}: Found {len(facesCurFrame)} faces")
            total_faces_processed += len(facesCurFrame)
            
            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                
                if len(faceDis) > 0:
                    matchIndex = np.argmin(faceDis)
                    conf_threshold = 0.6
                    
                    if matches[matchIndex] and faceDis[matchIndex] < conf_threshold:
                        name = classNames[matchIndex].upper()
                        print(f"Found {name} in {img_name} (Confidence: {1-faceDis[matchIndex]:.2f})")
                        if mark_attendance(name, output_file):
                            total_attendance_marked += 1
                    else:
                        print(f"Unknown face found in {img_name}")
            
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
    
    print(f"\nProcessing Summary:")
    print(f"Total images processed: {total_images}")
    print(f"Total faces detected: {total_faces_processed}")
    print(f"Total attendance marked: {total_attendance_marked}")

if __name__ == "__main__":
    # Define paths
    REFERENCE_IMAGES_PATH = 'Images'  # Folder with reference images
    INPUT_FOLDER = 'Input'  # Folder with new images to process
    OUTPUT_FILE = 'Attendance.csv'  # Output attendance file
    
    # Create Input folder if it doesn't exist
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"Created {INPUT_FOLDER} folder. Please add images to process.")
    
    # Process images
    print(f"Processing images from {INPUT_FOLDER}...")
    process_input_images(INPUT_FOLDER, REFERENCE_IMAGES_PATH, OUTPUT_FILE)
    print("Processing complete. Check Attendance.csv for results.") 