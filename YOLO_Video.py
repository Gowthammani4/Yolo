from ultralytics import YOLO
import cv2
import math
import numpy as np
from PIL import Image
import io
import base64
from pymongo import MongoClient
def video_detection(path_x):
    video_capture = path_x
    #Create a Webcam Object
    cap=cv2.VideoCapture(video_capture)
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))
    classNames={0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 
    8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
     29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 
     35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
     40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 
     47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
      53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 
      59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 
      65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
       70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
    model=YOLO("yolov8n.pt")
    while True:
        success, img = cap.read()
        results=model(source=img,stream=True,)
        print(results)
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                print(x1,y1,x2,y2)
                cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                class_name=classNames[cls]
                label=f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)

        yield img
        # client = MongoClient('mongodb://localhost:27017')
        # db=client["flask_app"]
        # col=db["pred_images"]
        # cv2.imwrite("test_result.jpeg",img)
        # img=np.squeeze(img.render())
        # image = Image.open("test_result.jpeg")
        # image_byte_array = io.BytesIO()
        # image.save(image_byte_array, format='JPEG')
        # image_base64 = base64.b64encode(image_byte_array.getvalue()).decode('utf-8')
# Create document
        # document = {'image': image_base64}
# Insert document into collection
        # col.insert_one(document)
        
#         document = col.find_one()
# # Extract the image data
#         image_base64 = document['image']
# # Decode image from base64
#         image_bytes = base64.b64decode(image_base64)

# # Load the image using PIL
#         image = Image.open(io.BytesIO(image_bytes))
#         cv2.imwrite("mongodb_img",image)
        

        #out.write(img)
        # cv2.imshow("image", img)
        #if cv2.waitKey(1) & 0xFF==ord('1'):
            #break
    #out.release()
cv2.destroyAllWindows()
# if __name__ == "__main__":


