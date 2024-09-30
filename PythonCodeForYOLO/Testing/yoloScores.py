from ultralytics import YOLO
import time
import pandas as pd

def getInferenceTime(modelName, nImage): 
    model = YOLO(modelName)
    model.to("cuda") # move the model to GPU
    startTime = time.time() # Starting clock
    model.predict("Datasets/MergedParkingDataset.v1i.yolov8/valid/images/", verbose = False)# source for all the images
    endTime = time.time()
    totalTime = endTime - startTime # time elasped
    avgTime = totalTime/nImage # avg time elasped (in sec)
    avgTimeInMs = avgTime*1000 # avg time elasped (in ms)
    return avgTimeInMs

# Model names 
"""
runs/detect/train_n/weights/best.pt
runs/detect/train_s/weights/best.pt
runs/detect/train_m/weights/best.pt
runs/detect/train_l/weights/best.pt
runs/detect/train_x/weights/best.pt
"""
if __name__ == '__main__':
    #intialising average Times
    n_timeAvg = 0
    s_timeAvg = 0
    m_timeAvg = 0
    l_timeAvg = 0
    x_timeAvg = 0
        
    for i in range(10): 
        # note need to get cashe and GPU warmed up before running time tests else data gets skewed
        n_time = getInferenceTime("runs/detect/train_n/weights/best.pt",nImage=46)
        s_time = getInferenceTime("runs/detect/train_s/weights/best.pt",nImage=46)
        m_time = getInferenceTime("runs/detect/train_m/weights/best.pt",nImage=46)
        l_time = getInferenceTime("runs/detect/train_l/weights/best.pt",nImage=46)
        x_time = getInferenceTime("runs/detect/train_x/weights/best.pt",nImage=46)
        if (i > 4): # The last 5 will be used for the average
            # This allows for the cashe to warm up then start calculating averages
            n_timeAvg = n_timeAvg +n_time/5 
            s_timeAvg = s_timeAvg +s_time/5 
            m_timeAvg = m_timeAvg +m_time/5 
            l_timeAvg = l_timeAvg +l_time/5 
            x_timeAvg = x_timeAvg +x_time/5  
            print(i)
    FOMO_time = 26
    modelNames = ["YoloV8n", 'YoloV8s', "FOMO",'YoloV8m',  'YoloV8l', 'YoloV8x']
    modelTimes = [n_timeAvg, s_timeAvg, FOMO_time,m_timeAvg, l_timeAvg, x_timeAvg]
    data = {"ModelName": modelNames, "ModelTimes":modelTimes}
    df = pd.DataFrame(data)
    print(df)
    df.to_csv('SpeedData.csv', index=False)