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
        

        return self.train_dataset, self.val_dataset, self.test_dataset




    def training(self,train_dataset, val_dataset, test_dataset):
        
        with open("parameters.yaml", "r") as f:
         config = yaml.safe_load(f)
        
        config['train_dataset'] = train_dataset
        config['val_dataset'] = val_dataset
        config['test_dataset'] = test_dataset
        config['collate_fn'] = self.collate_fn


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