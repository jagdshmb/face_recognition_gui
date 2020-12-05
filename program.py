import tkinter as tk
import cv2
import os
import PIL
from PIL import Image
import numpy as np
import pickle

window = tk.Tk()
window.geometry('400x200')
heading = tk.Label(window,text="FACE RECOGNITION SYSTEM")
heading.pack()

def TakeImages():
    name = input("Enter your name : ")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir,"trainimages")
    folder_dir = os.path.join(image_dir,name)
    os.mkdir(folder_dir)

    detector = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
    cap = cv2.VideoCapture(0)
    number = 0
    while(True):
        ret,image = cap.read()
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray,1.5,5)
        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            number = number+1
            os.chdir(folder_dir)
            cv2.imwrite(str(number)+".png",gray[y:y+h,x:x+w])
            os.chdir(base_dir)
        cv2.imshow('frame',image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        elif number > 100:
            break

    cap.release()
    cv2.destroyAllWindows()
    temp = "Data images of "+name+" captured"
    message1.configure(text = temp)
                    
def TrainImages():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, "trainimages")

    detector = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    current_id = 0
    label_ids = {}
    y_labels = []
    x_train = []

    for root,dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
                if label in label_ids:
                    pass
                else:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]
                print(label_ids)
                pil_image = Image.open(path).convert("L") # grayscale
                image_array = np.array(pil_image, "uint8")
                faces = detector.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
                for(x,y,w,h) in faces:
                    roi = image_array[y:y+h ,x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)
    print(y_labels)
    print(x_train)
    with open("labels.pickle", 'wb') as f:
        pickle.dump(label_ids , f)

    recognizer.train(x_train , np.array(y_labels))
    recognizer.save("trainner.yml")
    temp = "Training of Data images completed"
    message2.configure(text = temp)    

def IdentifyFaces():
    detector = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("./trainner.yml")

    labels = {"person_name":1}
    with open("labels.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v:k for k,v in og_labels.items()}

    cap = cv2.VideoCapture(0)

    while(True):
        ret,frame = cap.read()
        if not frame is None:
            if not ret: continue
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray,1.5,5)
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h,x:x+w]
            id_ ,conf = recognizer.predict(roi_gray)
            if conf >= 45:
                print(id_)
                print(labels[id_])
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (255,255,255)
                stroke = 2
                cv2.putText(frame, name ,(x,y), font, 1, color, stroke, cv2.LINE_AA)
                color = (255,0,0)
                stroke = 2
                cv2.rectangle(frame,(x,y),(x+w,y+w),color,stroke)
        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
            

    cap.release()
    cv2.destroyAllWindows()

b1 = tk.Button(window,text="Take Images",command=TakeImages)
b1.pack()
message1 = tk.Label(window,text="")
message1.pack()
b2 = tk.Button(window,text="Train Images",command=TrainImages)
b2.pack()
message2 = tk.Label(window,text="")
message2.pack()
b3 = tk.Button(window,text="Identify Faces",command=IdentifyFaces)
b3.pack()


