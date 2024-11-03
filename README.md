

### Python Code Example

```python
import face_recognition
import cv2
import os
import numpy as np

# Step 1: Load known faces of missing persons
def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []

    for file_name in os.listdir(known_faces_dir):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            # Load and encode the image
            image_path = os.path.join(known_faces_dir, file_name)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)

            if encoding:  # If at least one face was found
                known_face_encodings.append(encoding[0])
                # Use filename (without extension) as the name
                known_face_names.append(os.path.splitext(file_name)[0])

    return known_face_encodings, known_face_names

# Path to folder with images of missing persons
known_faces_dir = "path_to_missing_person_images"  # Replace with actual path
known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

# Step 2: Define function to process video frames
def find_missing_person_in_frame(frame, known_face_encodings, known_face_names):
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Initialize array to store names of matched persons
    face_names = []
    
    for face_encoding in face_encodings:
        # Compare detected face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use face distance to find the best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:  # If match found
            name = known_face_names[best_match_index]
        
        face_names.append(name)

    # Annotate frame with detected names
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Draw label with name below face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    return frame

# Step 3: Run the detection on live video or CCTV footage
video_capture = cv2.VideoCapture(0)  # Use video file path or 0 for webcam

while True:
    # Capture a single frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Process frame to find missing persons
    processed_frame = find_missing_person_in_frame(frame, known_face_encodings, known_face_names)

    # Display the processed frame
    cv2.imshow('Video - Missing Person Detection', processed_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close display windows
video_capture.release()
cv2.destroyAllWindows()
```
