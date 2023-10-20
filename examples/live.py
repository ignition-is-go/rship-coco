import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np
from supervision import ColorLookup
import os


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture('/home/trevor/Downloads/pexels-raquel-tinoco-6477058 (1080p).mp4')

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    absFilePath = os.path.abspath(__file__)

    # models folder path next to this file

    modelsFolderPath = os.path.dirname(absFilePath) + "/models"

    print(modelsFolderPath)
    exit()
    model = YOLO("./models/yolov8x.pt")

    model.to('cuda')


    while True:
        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind video
            continue

        results = model(frame)[0]

        detections = sv.Detections.from_ultralytics(results)

        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        labels = [
            results.names[class_id]
            for class_id
            in detections.class_id
        ]

        annotated_image = bounding_box_annotator.annotate(
            scene=frame, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels)

        cv2.imshow("yolov8", annotated_image)

        if cv2.waitKey(1) == 27:
            break


if __name__ == "__main__":
    main()
