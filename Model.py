import supervision as sv 
import cv2
import torch
from rfdetr import RFDETRBase
import yaml


""" 
RFDETR model

Backbone → Feature extractor
Neck/transformer → DETR-style attention module
Head → Prediction of bounding boxes and class labels
"""


class RFDETR():

    def __init__(self):
        self.model = RFDETRBase(pretrained=True)

        
    def training(self):
        
        with open("parameters.yaml", "r") as f:
         config = yaml.safe_load(f)

        self.trained_model = self.model.train(**config)

        return self.trained_model


    def predict(self, img, threshold, trained):


        model_used = self.trained_model if trained else self.model
        assert model_used is not None

        detections = model_used.predict(
         img,
         threshold=threshold)

        # Create annotators
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5)

        # Annotate image
        annotated_img = box_annotator.annotate(
         scene=img.copy(),
         detections=detections # type: ignore
          )

        annotated_img = label_annotator.annotate(
         scene=annotated_img,
         detections=detections # type: ignore
          )
        # Save result
        output_path = r"C:\\Project\\images_annotated.jpg"
        cv2.imwrite(output_path, annotated_img)

        print(f"Annotated image saved to: {output_path}")

        return detections, annotated_img

        



    





