import PIL.Image
import numpy as np
import face_recognition

# Load a single test image
test_image = PIL.Image.open("Images/ayush.jpg").convert('RGB')
np_image = np.array(test_image)
print(f"Image shape: {np_image.shape}, dtype: {np_image.dtype}")

# Try face detection
faces = face_recognition.face_locations(np_image)
print(f"Found {len(faces)} faces")