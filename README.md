# Face Recognition Attendance System

This project implements a face recognition-based attendance system with two modes of operation:
1. **Real-time Webcam Processing**: Detects faces from a webcam and marks attendance in real-time
2. **Batch Image Processing**: Processes multiple images from a folder and marks attendance for all detected faces

The system uses Python with OpenCV and face_recognition libraries to detect and recognize faces.

## Features

### Real-time Webcam Processing
- Real-time face detection using webcam
- Automatic attendance marking with timestamps
- Optimized performance (processes every 2nd frame)
- Press 'q' to exit the program
- Visual feedback with face rectangles and names

### Batch Image Processing
- Process multiple images from an Input folder
- Detect and recognize multiple faces in each image
- Mark attendance for recognized students
- Prevent duplicate attendance entries for the same day
- Detailed processing statistics and feedback
- Confidence scores for face matches
- Automatic Input folder creation

## Requirements

- Python 3.11.5
- OpenCV (opencv-python==4.11.0.86)
- face_recognition (face-recognition==1.3.0)
- dlib (dlib==19.24.1)
- cmake (cmake==4.0.0)

## Project Structure

```
.
├── Images/              # Folder containing reference images of students
├── Input/              # Folder for new images to process (batch mode)
├── Attendance.csv      # Output file for attendance records
├── AttendanceProject.py # Script for real-time webcam processing
├── process_images.py   # Script for batch image processing
└── README.md           # This file
```

## Setup

1. Create a folder named `Images` and add reference images of students
   - Each image should be named with the student's name (e.g., "John_Doe.jpg")
   - Images should be clear and show the student's face
   - Used by both real-time and batch processing modes

2. For batch processing:
   - The `Input` folder will be created automatically when you run the script
   - Place all images you want to process in this folder

## Usage

### Real-time Webcam Processing

1. Run the real-time processing script:
   ```bash
   python AttendanceProject.py
   ```

2. The script will:
   - Access your webcam
   - Detect faces in real-time
   - Display the video feed with face rectangles and names
   - Mark attendance for recognized faces
   - Press 'q' to exit the program

3. Features:
   - Processes every 2nd frame to reduce lag
   - Shows confidence scores for face matches
   - Prevents duplicate attendance entries
   - Works in real-time with visual feedback

### Batch Image Processing

1. Run the batch processing script:
   ```bash
   python process_images.py
   ```

2. The script will:
   - Load and encode all reference images
   - Process each image in the Input folder
   - Detect and recognize faces in each image
   - Mark attendance for recognized students
   - Generate a processing summary

3. Check `Attendance.csv` for the results
   - Each entry contains: Name, Time, Date
   - Duplicate entries for the same student on the same day are prevented

## Output Format

### Real-time Processing Output
- Live video feed with face detection
- Names and confidence scores displayed on screen
- Attendance marked automatically in Attendance.csv

### Batch Processing Output
1. For each image:
   - Number of faces detected
   - Names of recognized students
   - Confidence scores for each match

2. Final summary:
   - Total images processed
   - Total faces detected
   - Total attendance records marked

## Example Output

### Real-time Processing
```
Loading reference images...
Successfully loaded John_Doe.jpg
Successfully loaded Jane_Smith.jpg
Starting webcam...
[Press 'q' to exit]
```

### Batch Processing
```
Loading reference images...
Successfully loaded John_Doe.jpg
Successfully loaded Jane_Smith.jpg

Starting to process 3 images from Input...

Processing image1.jpg: Found 2 faces
Found JOHN_DOE in image1.jpg (Confidence: 0.85)
Found JANE_SMITH in image1.jpg (Confidence: 0.92)

Processing image2.jpg: Found 1 face
Found JOHN_DOE in image2.jpg (Confidence: 0.88)

Processing image3.jpg: Found 3 faces
Found JANE_SMITH in image3.jpg (Confidence: 0.90)
Unknown face found in image3.jpg
Found JOHN_DOE in image3.jpg (Confidence: 0.87)

Processing Summary:
Total images processed: 3
Total faces detected: 6
Total attendance marked: 4
```

## Notes

- The system uses a confidence threshold of 0.6 for face matching
- Images should be well-lit and show clear faces for best results
- The system can handle multiple faces in a single image
- Attendance is marked only once per student per day
- For real-time processing, ensure good lighting and clear view of faces
- For batch processing, ensure images are of good quality and show clear faces

## 🤝 Contributing
Feel free to **fork this repository** and submit a pull request with improvements!

## 📜 License
This project is **open-source** and free to use.

---
📧 **Need help?** Open an issue or contact me at [gargayush970@gmail.com]