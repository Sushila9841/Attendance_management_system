import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import os
import datetime
# Load the dataset
attendance_log="attendance.txt"
data_dir = "dataset_images"
label_file = "label.txt"

# Read the labels from label.txt
labels = []
with open(label_file, "r") as file:
    lines = file.readlines()
    for line in lines:
        label = line.strip().split(" ")[1]
        labels.append(label)

# Initialize the data and labels arrays
data = []
labels_encoded = []

# Iterate over the dataset folders
for label_idx, folder in enumerate(os.listdir(data_dir)):
    folder_path = os.path.join(data_dir, folder)
    # Iterate over the images in the folder
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        # Read and preprocess the image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))  # Resize to desired input shape
        img = img / 255.0  # Normalize pixel values
        data.append(img)
        labels_encoded.append(label_idx)

# Convert data and labels to numpy arrays
data = np.array(data)
labels_encoded = np.array(labels_encoded)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

# Define the model architecture
model = keras.Sequential([
    keras.applications.MobileNetV2(include_top=False, input_shape=(224, 224, 3)),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(len(labels), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save("facedetection.h5")

def get_class_name(class_no):
    if class_no == 0:
        return "Rajiv"
    elif class_no == 1:
        return "Phushan Thapa Magar"
    elif class_no == 2:
        return "Rabin"
    elif class_no == 3:
        return "Pratik"
    elif class_no == 4:
        return "Deepak"
    elif class_no == 5:
        return "Rovika"
    
model = keras.models.load_model('facedetection.h5')

# ================for test data===================
# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX

while True:
    success, img_original = cap.read()
    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img_original, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)  # Draw rectangle around face
        crop_img = img_original[y:y + h, x:x + w]
        img = cv2.resize(crop_img, (224, 224))
        img = img / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        class_name = get_class_name(class_index)

        cv2.putText(img_original, class_name,
                    (x, y + h + 20), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

        # Register attendance in the log file
        with open(attendance_log, "a") as file:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            attendance_entry = f"{timestamp} - {class_name}\n"
            file.write(attendance_entry)

        # Display attendance registration message
        cv2.putText(img_original, "Attendance Registered",
                    (10, 30), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Result", img_original)
    cv2.waitKey(1)

    if len(faces) > 0:
        # Wait for 3 seconds (3000 milliseconds) after face detection
        cv2.waitKey(3000)
        break

cap.release()
cv2.destroyAllWindows()
