# SwiftMark
### Project Description  Our AI-based attendance system leverages deep learning, using a CNN model (VGG16), to streamline the attendance marking process. Primarily designed for educational institutions, it recognizes faces from captured images to efficiently record attendance, significantly reducing manual effort and time.

### Features
- **Face Capturing:** Capture images using a camera.
- **Face Recognition:** Recognize faces from captured images using a trained CNN model.
- **Attendance Marking:** Automatically mark attendance for recognized faces.

### Technology Stack
- **Programming Language:** Python
- **Frameworks and Tools:** Google Colab
- **AI/ML Models:** CNN (VGG16)

### File Descriptions
- **FaceCapture:** Script that captures 100 images of each student.
- **FaceRecon:** Script for face recognition using the trained model.
- **haarcascade_frontalface_default.xml:** Pre-trained XML file for detecting frontal faces using the Haar Cascade classifier.
- **face_dataset_dict:** A list of faces indexed, used for mapping recognized faces to student identities.
- **MarkAttendance:** Script that marks attendance for recognized students.
- **face_recog_vgg.h5:** Version 1 of the trained CNN model.
- **face_recog_vgg_v2.h5:** Version 2 of the trained CNN model with improved accuracy.

### Challenges and Future Enhancements
- Improve data storage and security.
- Develop a user-friendly interface.
