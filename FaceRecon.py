import cv2 # type: ignore
import numpy as np # type: ignore
from tensorflow.keras.models import load_model # type: ignore
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

def read_class_labels(file_path):
    class_labels = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split(']: ')
            key = int(key.strip('['))
            class_labels[key] = value
    return class_labels

class_labels = read_class_labels('face_dataset_dict.txt') 

def face_recon():
    print("Starting camera")
    cap = cv2.VideoCapture(0)
    print("Camera Started")

    while True:
        success, frame = cap.read()
        print("Camera Read")
        
        if not success:
            print("No Success")
            break
        
        face, co_ordinates = fd.face_extractor_and_coordinates(frame)
        
        print("Face Extracted")
        
        if face is not None:
            predicted_label = predict_class(face)
            (x, y, w, h) = co_ordinates
        else:
            predicted_label = "Face not found"
            (x, y, w, h) = (0,0,0,0)
        
        print("face predicted")
        
        cv2.putText(frame, predicted_label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('Video', frame)
        cv2.waitKey(16) 

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_recon()
