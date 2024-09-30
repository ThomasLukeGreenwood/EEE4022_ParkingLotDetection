"""
This file is to test a single image

"""

from ultralytics import YOLO
import cv2

#resizing the image
# Other image optionst

#1-3 are the different parking lots from the test set 4 is from an untrained parking lot


#img = "Datasets/MergedParkingDataset.v1i.yolov8/test/images/photo_6_jpg.rf.38deb508061aee7638aa91b503c9f1b1.jpg" # Test data
#img = "Datasets/MergedParkingDataset.v1i.yolov8/test/images/4k-time-lapse-car-parking-lot-stock-video-download-video-clip-now-istock_TyROSAGZ_mp4-18_jpg.rf.29711c038857f37764e52a4f800a52eb.jpg"
img = "UntrainedData/SP0.jpeg"
#img = "Datasets/MergedParkingDataset.v1i.yolov8/test/images/photo_42_jpg.rf.e27e6bc9bab5f5d312a155981c4482e8.jpg"

#img = "UntrainedData/FromWeb.jpg"

imgSx,imgSy,imgRBG = (cv2.imread(img)).shape

print("Width = ", imgSx)    
print("Height = ", imgSy)

imgRs =  cv2.resize(cv2.imread(img), (640,640))
# Load a model
modelName = "runs/detect/train_m/weights/best.pt"

model = YOLO(modelName)  # pretrained YOLOv8n model
model.to("cuda")

results = model.predict(source = imgRs, show = True, show_labels = False, conf = 0.3, save = False, iou = 0.7) # can add Visualise = true for useful data ,imgsz = (imgSx,imgSy),
while (True):
    if cv2.waitKey(25)& 0xFF == ord('q'): # Break out of loop when q is pressed
        break   
cv2.destroyAllWindows() # Stop displaying the frame 
