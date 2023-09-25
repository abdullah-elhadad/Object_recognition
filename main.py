from tkinter import *
from tkinter import filedialog as tkFileDialog
import cv2


def ImgFile():
    path = tkFileDialog.askopenfilename()
    img = cv2.imread(path)
    classNames = []
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightpath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightpath, configPath)
    net.setInputSize(320, 230)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    print(classIds, bbox)
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
        cv2.putText(img, f"{classNames[classId - 1]},{confidence:.2f}%", (box[0] + 10, box[1] + 20),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)
    cv2.imshow('Output', img)
    cv2.waitKey(0)


def Video():
    path = tkFileDialog.askopenfilename()
    cap = cv2.VideoCapture(path)

    classNames = []
    classFile = 'coco.names'

    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightpath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightpath, configPath)
    net.setInputSize(320, 230)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    # Read until video is completed
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        classIds, confs, bbox = net.detect(frame, confThreshold=0.5)
        print(classIds, bbox)
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, f"{classNames[classId - 1]},{confidence:.2f}%", (box[0] + 10, box[1] + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to  exit
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


def Camera():
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)

    classNames = []
    classFile = 'coco.names'

    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightpath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightpath, configPath)
    net.setInputSize(320, 230)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    while True:
        success, img = cam.read()
        img = cv2.flip(img,1)
        classIds, confs, bbox = net.detect(img, confThreshold=0.5)
        print(classIds, bbox)
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, f"{classNames[classId - 1]},{confidence:.2f}%", (box[0] + 10, box[1] + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)
        cv2.imshow('Output', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
# ImgFile()
# Camera()
myframe = Tk()
myframe.title("APP")
myframe.geometry("500x400")
myframe.configure(background="black")
mylable = Label(myframe, text="OBJECT RECOGNITION", fg="#b8a81d", bg="#102e4f",
                font="Kartika", pady=15, padx=15)
mylable.pack()
mybutton_1 = Button(myframe, text="Click here for detect by image", fg="blue", bg="white",
                    font="Kartika", pady=14,activebackground="green", command=ImgFile)
mybutton_2 = Button(myframe, text="Click here for detect by stream video", fg="blue", bg="white",
                    font="Kartika", pady=14,activebackground="green", command=Camera)
mybutton_3 = Button(myframe, text="Click here for detect by video", fg="blue", bg="white",
                    font="Kartika", pady=14,activebackground="green", command=Video)
mybutton_4 = Button(myframe, text="exit", fg="red", bg="white",
                    font="Kartika", pady=8, padx=30,activebackground="red", command=exit)
mybutton_1.pack()
mybutton_2.pack()
mybutton_3.pack()
mybutton_4.pack()
myframe.mainloop()