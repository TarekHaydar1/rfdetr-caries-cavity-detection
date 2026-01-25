import torch
from Model import RFDETR
import cv2


def camera_feed_with_detection():
      return
      
       
if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    model = RFDETR()

    mode = input("\n\n1. Train the model \n2. Run prediction \n3. AI Camera feed\n4. Exit\nSelect mode: ")
    
    

    if mode == "1":
     trained_model = model.training()

    if mode == "2":
     
     #Prediction on a single image
     # Load image
     img = cv2.imread(r"C:\\Project\\images.jpg")
    
     detections, annotated_img = model.predict(
        img, threshold=0.5, trained=False)
     






    if mode == "4":
     exit()
    exit() 