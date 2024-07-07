from ultralytics import YOLO
import cv2 
from cvzone import putTextRect
import math
from sort import *
model = YOLO('yolo-Weights/yolov8n.pt')

# could you larger weights but my CPU is slow 
cap = cv2.VideoCapture("videos/cars.mp4")
cap.set(3, 1280)
cap.set(4, 720)

# COCO classes
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

mask = cv2.imread("Images/mask.png")
# change the shape of mask 
mask = cv2.resize(mask, (1280,720))

#using sort to track the object 
tracker = Sort(max_age=10, min_hits=3)

limit1 = ((130, 250), (640, 300)) #adjust by hand
limit2 = ((640, 300), (1150,250)) #adjust by hand

count0  = [] 
count2 = []
def dis(x,p1,p2):
    # check if x is in the region
    if (min(p1[0],p2[0])< x[0] <max(p1[0],p2[0])) and (min(p2[1],p1[1]) < x[1] <max(p1[1],p2[1])):
        # calculate the distance between a point and a line
        x1,y1 = p1
        x2,y2 = p2
        return abs((y2-y1)*x[0] - (x2-x1)*x[1] + x2*y1 - x1*y2) / math.sqrt((y2-y1)**2 + (x2-x1)**2) # highschool math bros
    else:
        return math.inf

i = 0 #for skipping frames
while True:
    i+=1
    ret, img = cap.read()
    if i%2 == 0:
        continue #skip frame because my CPU is slow
    imgRegion = cv2.bitwise_and(img,mask)
    results = model(imgRegion,stream=True) 
    
    detection = np.empty((0,5))
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255,0, 255), 3)
            
            conf = math.ceil(box.conf[0]*100)/100
            
            cls = int(box.cls[0])
            
            class_name = class_names[cls]
            label = f'{class_name}: {conf:.2f}'
            
            if class_name in ["car","truck","bus","motorcycle","bicycle"] and conf > 0.5:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255,0, 255), 3)
                # cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                current_detection = np.array([x1,y1,x2,y2,conf])
                detection = np.vstack((detection,current_detection))

    cv2.line(img, limit1[0], limit1[1], (0, 255, 0), 2) 
    cv2.line(img, limit2[0], limit2[1], (255, 0, 0), 2) 
    
    resultTracker = tracker.update(detection)
    for result in resultTracker: 
        x1,y1,x2,y2,id = result

        # label = f'ID: {id}'
        # cv2.putText(img, label , (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 125, 124), 2)

        cx, cy = map(int,[(x1+x2)//2, (y1+y2)//2])
        cv2.circle(img, (cx,cy), 5, (0, 0, 240), cv2.FILLED)
        

        if dis((cx,cy),limit1[0],limit1[1]) < 10:
            if count0.count(id) == 0:
                count0.append(id)
                
            
        if dis((cx,cy),limit2[0],limit2[1]) < 10:
            if count2.count(id) == 0:
                count2.append(id)

    # show the counts 
    cv2.putText(img, f'Count 1: {len(count0)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f'Count 2: {len(count2)}', (1280-200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imshow('img', img)
    # cv2.imshow('img region', imgRegion)
    
    if cv2.waitKey(1) == ord('q'): ## press q to break
        break