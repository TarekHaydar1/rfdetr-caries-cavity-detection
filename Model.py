import supervision as sv 
import cv2
from rfdetr import RFDETRBase
import yaml
from torchvision import transforms as T
from torchvision.datasets import CocoDetection
from rfdetr.datasets.coco import make_coco_transforms

""" 
RFDETR model

Backbone → Feature extractor
Neck/transformer → DETR-style attention module
Head → Prediction of bounding boxes and class labels
"""


class RFDETR():


    def __init__(self):
        self.model = RFDETRBase(pretrained=True)


    def collate_fn(self, batch):

     return tuple(zip(*batch))


    def dataset_preparation(self):
        
        train_transforms = make_coco_transforms(image_set="train",
                                                resolution=672,
                                                multi_scale=True,           
                                                expanded_scales=True,       
                                                skip_random_resize=False)
        
        val_transforms = make_coco_transforms(image_set="val",
                                                resolution=672)
        
        test_transforms = make_coco_transforms(image_set="val",
                                                resolution=672)                
        
        try:
         self.train_dataset = CocoDetection(
         root=r"C://Project//dataset//train",
         annFile=r"C://Project//dataset//train//_annotations.coco.json",
         transform=train_transforms)

         self.val_dataset = CocoDetection(
         root="C://Project//dataset//valid",
         annFile="C://Project//dataset//valid//_annotations.coco.json",
         transform=val_transforms)

         self.test_dataset = CocoDetection(
         root="C://Project//dataset//test",
         annFile="C://Project//dataset//test//_annotations.coco.json",
         transform=test_transforms)
         
        except Exception as e:
         self.train_dataset = CocoDetection(
         root=r"/content/drive/MyDrive/Dataset.coco/train/train",
         annFile=r"/content/drive/MyDrive/Dataset.coco/train/_annotations.coco.json",
         transform=train_transforms)

         self.val_dataset = CocoDetection(
         root=r"/content/drive/MyDrive/Dataset.coco/valid/valid",
         annFile=r"/content/drive/MyDrive/Dataset.coco/valid/_annotations.coco.json",
         transform=val_transforms)

         self.test_dataset = CocoDetection(
         root=r"/content/drive/MyDrive/Dataset.coco/test/test",
         annFile=r"/content/drive/MyDrive/Dataset.coco/test/_annotations.coco.json",
         transform=test_transforms)


        return self.train_dataset, self.val_dataset, self.test_dataset




    def training(self,train_dataset, val_dataset, test_dataset):
        
        try:
         dataset_dir= "C:\\Project\\dataset"
         output_dir= "C:\\Project\\train_output"
         with open("parameters.yaml", "r") as f:
          config = yaml.safe_load(f)

        except Exception as e:
         dataset_dir= "/content/drive/MyDrive/Dataset.coco"
         output_dir= "/content/drive/MyDrive/train_output"
         with open("/content/rfdetr-caries-cavity-detection/Parameters.yaml", "r") as f:
          config = yaml.safe_load(f)
        
       

        self.trained_model = self.model.train(train_dataset=train_dataset,
                                              val_dataset=val_dataset,
                                              test_dataset=test_dataset,
                                              collate_fn=self.collate_fn,
                                              dataset_dir= dataset_dir,
                                              output_dir= output_dir,
                                              **config)

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
        try:
         output_path = r"C:\\Project\\images_annotated.jpg"
        except Exception as e:
         output_path = r"/content/rfdetr-caries-cavity-detection/images_annotated.jpg"
        
        cv2.imwrite(output_path, annotated_img)

        print(f"Annotated image saved to: {output_path}")

        return detections, annotated_img