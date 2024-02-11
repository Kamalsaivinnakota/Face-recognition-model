import cv2
import face_recognition
import pickle
from datetime import datetime, timedelta
import csv

# Dictionary to store the last recognition time for each person
last_recognition_time = {}

# Time interval for recording attendance for the same person (in minutes)
RECOGNITION_INTERVAL_MINUTES = 10

def recognize_faces(frame, knn_model_path):
    global last_recognition_time
    
    # Load the trained KNN classifier
    with open(knn_model_path, 'rb') as f:
        knn_clf = pickle.load(f)

    # Find all face locations in the frame
    face_locations = face_recognition.face_locations(frame)

    # If no faces are found, return an empty list
    if not face_locations:
        return []

    # Get face encodings for all faces in the frame
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Use the KNN classifier to predict the names of the recognized faces
    names = knn_clf.predict(face_encodings)

    # Get current timestamp
    current_time = datetime.now()

    # List to store recognized faces with timestamps
    recognized_faces = []

    # Iterate through recognized faces
    for face_location, name in zip(face_locations, names):
        # Check if the person has been recognized within the specified interval
        if name in last_recognition_time:
            elapsed_time = current_time - last_recognition_time[name]
            if elapsed_time < timedelta(minutes=RECOGNITION_INTERVAL_MINUTES):
                continue  # Skip recording attendance

        # Record the face recognition
        recognized_faces.append((face_location, name, current_time))  
        last_recognition_time[name] = current_time  # Update last recognition time

    return recognized_faces

def store_attendance(attendance_data, csv_filename):
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        for entry in attendance_data:
            writer.writerow([entry[0], entry[1], entry[2].strftime("%Y-%m-%d %H:%M:%S")])

def draw_boxes(frame, recognized_faces):
    # Clear the frame before drawing new bounding boxes
    frame_copy = frame.copy()

    # Iterate through recognized faces list
    for (top, right, bottom, left), name, timestamp in recognized_faces:
        # Draw the box around the face
        cv2.rectangle(frame_copy, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw the name label above the face
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame_copy, f'{name} ({timestamp.strftime("%Y-%m-%d %H:%M:%S")})', (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the frame with boxes and names
    cv2.imshow("Recognized Faces", frame_copy)

if __name__ == "__main__":
    # Path to the trained KNN model
    knn_model_path = "trained_knn_model.clf"

    # CSV filename to store attendance data
    csv_filename = "attendance.csv"

    # Open the webcam
    cap = cv2.VideoCapture(0)  # Change the index if your webcam is not at index 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Recognize faces in the frame
        recognized_faces = recognize_faces(frame, knn_model_path)

        # Draw boxes and names on the frame
        draw_boxes(frame, recognized_faces)

        # Store attendance data in the CSV file
        store_attendance(recognized_faces, csv_filename)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close any open windows
    cap.release()
    cv2.destroyAllWindows()
