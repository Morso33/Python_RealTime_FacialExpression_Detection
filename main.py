import cv2, os
import numpy as np
from PIL import Image

faceTypes = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
face_images = []
face_labels = []


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
        for filename in os.listdir("faces/" + faceTypes[i] + "/"):
            print(filename)
            #Skip over 9/10 of the images to speed up training
            if (np.random.randint(0, 10) > 0):
                continue
            
            #Does the file exist?
            if not os.path.isfile("faces/" + str(faceTypes[i]) + "/" + filename):
                print ("skipped file: faces/" + str(faceTypes[i])+ "/" + filename)
                continue
            image = cv2.imread("faces/"+ str(faceTypes[i]) + "/" + filename, cv2.IMREAD_GRAYSCALE)
            if (image is None):
                print("image is none: " + "faces/" + filename)
                continue
            image = cv2.resize(image, (100, 100))
            face_images.append(image)
            face_labels.append(i)
            print(str(filename))

    print("Creating model...")    
    # Create the LBPH model
    face_images = np.array(face_images)
    face_labels = np.array(face_labels)
    model = cv2.face.LBPHFaceRecognizer_create()
    # Train the model on the face images and labels
    model.train(face_images, face_labels)
    print("Model created!")
    #Save the model
    model.save("model.yml")
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


    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    # Loop over each detected face
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (100, 100))
        label, confidence = model.predict(roi_gray)

        # Display the emotion label on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_text = "Emotion: " + str(faceTypes[label])
        cv2.putText(gray, label_text, (x, y-10), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        #Display the confidence
        confidence_text = "Confidence: " + str(confidence)
        cv2.putText(gray, confidence_text, (x, y-30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        

        # Draw a rectangle around the face
        cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Frame', gray)
    if cv2.waitKey(1) == 27: #ESC
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
