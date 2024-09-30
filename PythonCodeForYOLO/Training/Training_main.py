from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)
    model.to("cuda")
    #for dataset in this model
    #results = model.train(data="C:/Users/tlgwo/Documents/UCT/4th Year/EEE4022/TestingAndTrainingYOLO/dataset/data.yaml", epochs=100, imgsz=640)
    
    #For the other dataset
    results = model.train(data="C:/Users/tlgwo/Documents/UCT/4th Year/EEE4022_Final/Datasets/MergedParkingDataset.v1i.yolov8/data.yaml", epochs=50, imgsz=640)
    print("Training Complete!!!")

    