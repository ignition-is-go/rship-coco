import cv2
from ultralytics import YOLO
import supervision as sv
import torch

if torch.cuda.is_available():
    print('CUDA is available. PyTorch is using the GPU.')
else:
    print('CUDA is not available. PyTorch is using the CPU.')

model = YOLO("yolov8n.pt")
model.to('cuda')
image = cv2.imread(r"c:\Users\Lucid\SAM\man-apple.jpg")
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [
    results.names[class_id]
    for class_id
    in detections.class_id
]

annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels)

sv.plot_image(annotated_image)
