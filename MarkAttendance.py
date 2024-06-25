import cv2  # type: ignore
import numpy as np  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
import pandas as pd
import datetime
import FaceDetect as fd

print("Model Loading")
model = load_model('face_recog_vgg.h5')
print("Model Loaded")

# Function to predict class label given an input image
def predict_class(image):
    # Preprocess the image
    image_resized = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
    image_array = image_resized / 255.0
    image_input = np.expand_dims(image_array, axis=0)

    # Predict the class probabilities
    predictions = model.predict(image_input)
    predicted_class_idx = np.argmax(predictions)
    predicted_label = class_labels[predicted_class_idx]

    return predicted_label

# Function to read class labels from a file
def read_class_labels(file_path):
    class_labels = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split(']: ')
            key = int(key.strip('['))
            class_labels[key] = value
    return class_labels

class_labels = read_class_labels('face_dataset_dict.txt')

# Function to mark attendance in an Excel file
def mark_attendance(name):
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    attendance_file = f"Attendance_{date_str}.xlsx"

    try:
        attendance_df = pd.read_excel(attendance_file, sheet_name='Attendance')
    except FileNotFoundError:
        attendance_df = pd.DataFrame(columns=["Name", "Timestamp"])

    # Check if the student is already marked present for today
    if name not in attendance_df.values:
        attendance_df = attendance_df.append({"Name": name, "Timestamp": datetime.datetime.now().strftime("%H:%M:%S")}, ignore_index=True)
        attendance_df.to_excel(attendance_file, index=False, sheet_name='Attendance')
        print(f"Attendance marked for {name}")
    else:
        print(f"{name} is already marked present")

def face_recon():
    print("Starting camera")
    cap = cv2.VideoCapture(0)
    print("Camera Started")

    while True:
        success, frame = cap.read()

        if not success:
            print("Failed to read from camera")
            break

        face, co_ordinates = fd.face_extractor_and_coordinates(frame)

        if face is not None:
            predicted_label = predict_class(face)
            (x, y, w, h) = co_ordinates
            mark_attendance(predicted_label)
        else:
            predicted_label = "Face not found"
            (x, y, w, h) = (0, 0, 0, 0)

        cv2.putText(frame, predicted_label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_recon()
