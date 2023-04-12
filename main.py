import cv2, os
import numpy as np
from PIL import Image
import time

faceTypes = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
face_images = []
face_labels = []
cached_expression = "neutral"
cached_expression_last_updated = time.time()
last_detected_expression = "neutral"
last_detected_expression_time = time.time()

def cleanup_faces_folder():
    #If file is not a png, delete it
    for filename in os.listdir("faces/"):
        if not filename.endswith(".png"):
            os.remove("faces/" + filename)
            print("removed file: " + "faces/" + filename)

 

def create_model():
    global face_images
    global face_labels

    for i in range (0, len(faceTypes)):
        for j in range (0, 436): #We will train every expression with 430 examples, as disgusted only has 430 examples
            filename = "faces/" + faceTypes[i] + "/im" + str(j) + ".png"
            print(filename)
            if not (os.path.isfile(filename)):
                print("SKIP")
                continue

            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (100, 100))
            face_images.append(image)
            face_labels.append(i)

    print("Creating model...")    
    # Create the LBPH model
    face_images = np.array(face_images)
    face_labels = np.array(face_labels)
    model = cv2.face.LBPHFaceRecognizer_create()
    # Train the model on the face images and labels
    model.train(face_images, face_labels)
    print("Model created!")
    #Save the model
    print("Saving model...")
    model.save("model.yml")
    print("Model saved!")
    return model

if (os.path.isfile("model.yml")):
    print("Loading cached model...")
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read("model.yml")
    print("Model loaded!")
else:
    print("No cached model found, creating new model...")
    model = create_model()


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#Setup webcam
print("starting webcam, please wait...")
cap = cv2.VideoCapture(0)
print("ready!")
if not cap.isOpened():
    print("Error opening video stream")
# Loop until 'esc' is pressed
while True:
    ret, frame = cap.read()
    #Get the FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, "FPS: " + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #Get the gpu usage
    gpu_usage = cv2.cuda.getCudaEnabledDeviceCount()
    cv2.putText(frame, "GPU: " + str(gpu_usage), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #Get the webcam frame and convert it to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))
    # Loop over each detected face
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (100, 100))
        label, confidence = model.predict(roi_gray)

        if (faceTypes[label] != last_detected_expression):
            last_detected_expression = faceTypes[label]
            last_detected_expression_time = time.time()

        if (confidence > 93 and time.time() - cached_expression_last_updated > 0.15 and time.time() - last_detected_expression_time > 0.15): # If the detected expression has been static for one second, and 
            cached_expression = faceTypes[label]
            cached_expression_last_updated = time.time()

        # Display the emotion label and confidence score on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_text = "Emotion: " + cached_expression
        cv2.putText(gray, label_text, (x, y-10), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        confidence_text = "Confidence: {:.2f}".format(confidence)
        cv2.putText(gray, confidence_text, (x, y-30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        #Draw debug info top right
        cv2.putText(gray, "Detected: " + faceTypes[label], (450, 30), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(gray, "Confidence: {:.2f}".format(confidence), (450, 50), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(gray, "Cached: " + cached_expression, (450, 70), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(gray, "Last updated: " + str ( float(time.time()) - cached_expression_last_updated), (450, 90), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(gray, "Last detected: " + str ( float(time.time()) - last_detected_expression_time), (450, 110), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        # Draw a rectangle around the face
        cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Frame', gray)
    if cv2.waitKey(1) == 27: #ESC
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
