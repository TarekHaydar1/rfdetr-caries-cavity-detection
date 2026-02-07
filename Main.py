import torch
from Model import RFDETR
import cv2
import torch

def camera_feed_with_detection():
      return
      
       
if __name__ == "__main__":
    
  if torch.cuda.is_available():
    device = torch.device("cuda")  # GPU
  else:
    device = torch.device("cpu")   # CPU fallback
    
  model = RFDETR()
    
  mode = input("\n\n1. Train the model \n2. Run prediction \n3. Exit\nSelect mode: ")

  if mode == "1":
     train_dataset, val_dataset, test_dataset = model.dataset_preparation()
     trained_model = model.training(train_dataset, val_dataset, test_dataset)

  if mode == "2":
     
     #Prediction on a single image
     # Load image
     img = cv2.imread(r"C:\\Project\\images.jpg")
    
     detections, annotated_img = model.predict(
        img, threshold=0.5, trained=True)
     

  if mode == "3":
     exit()
  exit() 