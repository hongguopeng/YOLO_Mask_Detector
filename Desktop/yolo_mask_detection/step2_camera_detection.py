import cv2
import numpy as np
import imutils

net = cv2.dnn.readNetFromDarknet('yolov3.cfg' , 'yolov3.backup')
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
classes = [line.strip() for line in open('obj.names')]
colors = [(255 , 0 , 0) , (0 , 0 , 255) , (0 , 255 , 0)]

def yolo_detect(img):
    # forward propogation
    height , width , channels = img.shape
    blob = cv2.dnn.blobFromImage(img , 1/255. , (416 , 416) ,
                                 (0 , 0 , 0) , True , crop = False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    # get detection boxes
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            tx , ty , tw , th , confidence = detection[0:5]
            scores = detection[5:]
            prob = np.max(scores)
            class_id = np.argmax(scores)  
            if confidence > 0.3:   
                center_x = int(tx * width)
                center_y = int(ty * height)
                w = int(tw * width)
                h = int(th * height)
                # 取得box座標
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x , y , w , h , prob])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    # draw boxes
    indexes = cv2.dnn.NMSBoxes(boxes , confidences , 0.3 , 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x , y , w , h , prob = boxes[i]
            label = str(classes[class_ids[i]]) + ' {:.2%}'.format(prob)
            color = colors[class_ids[i]]
            cv2.rectangle(img , (x , y) , (x + w , y + h) , color , 2)
            cv2.putText(img , label , (x , y - 5) ,
                        cv2.FONT_HERSHEY_DUPLEX , 0.8 ,
                        color , 1 , cv2.LINE_AA)
    return img

VIDEO_IN = cv2.VideoCapture(0)
while True:
    hasFrame , frame = VIDEO_IN.read()
    img = yolo_detect(frame)
    cv2.imshow('Mask Detector' , imutils.resize(img , width = 1000))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
VIDEO_IN.release()
cv2.destroyAllWindows()
