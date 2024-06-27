import cv2 # type: ignore

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    if len(faces) == 0:
        return None
    for (x, y, w, h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    return cropped_face

def face_extractor_and_coordinates(img):
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None, None
    
    # Initialize variables to store the largest face found
    largest_face = None
    largest_area = 0
    face_coords = None
    
    for (x, y, w, h) in faces:
        area = w * h
        if area > largest_area:
            largest_face = img[y:y+h, x:x+w]
            face_coords = (x, y, w, h)
            largest_area = area
    
    return largest_face, face_coords    

def face_detector():
    print("Starting camera")
    cap = cv2.VideoCapture(0)
    print("Camera Started")
    
    while True:
        ret, frame = cap.read()
        print("Camera Read")
        
        if not ret:
            print("Failed to capture image")
            break
        
        face = face_extractor(frame)
        print("Face Extracted")
        
        if face is not None:
            face = cv2.resize(face, (128, 128))
            #face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            return face
        else:
            print("Face not found")
            break
    cap.release()
    cv2.destroyAllWindows()   
    return "Face not found"

def main():
    detected_face = face_detector()
    if detected_face is not None and isinstance(detected_face, str) and detected_face == "Face not found":
        print("Face not found in the frame.")
    elif detected_face is not None:
        print("Face detected and resized.")
        cv2.imshow("Detected Face", detected_face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No face detected.")

if __name__ == "__main__":
    main()
