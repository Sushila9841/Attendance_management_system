import cv2
import os

# Prompt the user to choose an option
option = input("Choose an option:\n1. Capture image from webcam\n2. Load image from file\n")

# Check the selected option
if option == '1':
    # Option 1: Capture image from webcam
    # Prompt the user to enter a name
    name = input("Enter your name: ")

    # Create a directory with the provided name
    directory = "dataset_images/{}".format(name)
    os.makedirs(directory, exist_ok=True)

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Counter for image filenames
    count = 1

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow("Capture", frame)

        # Wait for the 's' key to save the image
        if cv2.waitKey(1) & 0xFF == ord('s') and len(faces) > 0:
            # Save the captured image with incremental filenames
            image_path = "{}/{:02d}.jpg".format(directory, count)
            cv2.imwrite(image_path, frame)
            print("Image saved:", image_path)

            count += 1

        # Wait for the 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q') or count > 100:
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

elif option == '2':
    # Option 2: Load image from file
    # Prompt the user to enter the image file path
    file_path = input("Enter the image file path: ")

    # Check if the file exists
    if os.path.isfile(file_path):
        # Prompt the user to enter a name for the image folder
        name = input("Enter the folder name: ")

        # Create a directory with the provided name
        directory = "dataset_images/{}".format(name)
        os.makedirs(directory, exist_ok=True)

        # Load the image
        image = cv2.imread(file_path)

        # Save the loaded image with incremental filenames
        count = 1
        while count <= 100:
            image_path = "{}/{:02d}.jpg".format(directory, count)
            cv2.imwrite(image_path, image)
            print("Image saved:", image_path)
            count += 1

    else:
        print("File not found!")

else:
    print("Invalid option selected!")

# Generate label.txt
label_file = open("label.txt", "w")

# Get the list of folder names in the 'images' directory
folders = os.listdir("dataset_images")

# Write the labels to label.txt
for i, folder in enumerate(folders):
    label = "{} {}\n".format(i, folder)
    label_file.write(label)

# Close the label.txt file
label_file.close()

print("label.txt generated successfully!")