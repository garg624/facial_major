# Face Recognition Attendance System

This is a **real-time face recognition-based attendance system** that detects faces from a laptop webcam and marks attendance in a CSV file.

## 🚀 Features
- **Real-time face detection** using `face_recognition` and `OpenCV`
- **Automatic attendance marking** with timestamps
- **Press `q` to exit** the program anytime
- **Optimized for better performance** (processes every 2nd frame to reduce lag)

## 🛠️ Installation

Ensure you have **Python 3.x** installed, then run the following commands:

### 1️⃣ Install Dependencies
```sh
pip install cmake 
pip install .\dlib-19.24.1-cp311-cp311-win_amd64.whl
pip install opencv-python numpy face_recognition

```

### 2️⃣ Clone This Repository
```sh
git clone https://github.com/your-username/face-recognition-attendance.git
cd face-recognition-attendance
```

### 3️⃣ Create an `Images` Folder
Put the images of people you want to recognize inside the `Images/` folder.  
Each image should be named as the **person's name**, e.g., `John.jpg`, `Alice.png`.

### 4️⃣ Run the Program
```sh
python AttendanceProject.py
```

## 🎯 How It Works
1. The program **loads images** from the `Images/` folder.
2. It **encodes** the faces in the images for recognition.
3. The webcam **captures frames** and detects faces.
4. If a face matches, the **name is displayed** on the screen and **attendance is marked** in `Attendance.csv`.
5. **Press `q` to exit** the program.

## 📂 Project Structure
```
face-recognition-attendance/
│── Images/            # Folder containing images of known people
│── face_recognition_attendance.py   # Main Python script
│── Attendance.csv     # File where attendance is recorded
│── README.md          # Documentation
```

## 📝 CSV File Format (Attendance.csv)
```
Name,Time,Date
John,14:30:10,24/03/2025
Alice,14:32:45,24/03/2025
```

## 📌 Notes
- **Ensure images are clear and well-lit** for better recognition.
- **Run the script in a well-lit environment** for accurate face detection.
- If you experience lag, **reduce the frame rate** by modifying `frame_skip` in the script.

## 🤝 Contributing
Feel free to **fork this repository** and submit a pull request with improvements!

## 📜 License
This project is **open-source** and free to use.

---
📧 **Need help?** Open an issue or contact me at [gargayush970@gmail.com]