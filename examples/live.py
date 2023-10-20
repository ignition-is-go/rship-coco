import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np
from supervision import ColorLookup


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

    cap = cv2.VideoCapture(
        "rtsp://localhost/live")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("FastSAM-s.pt")
    model.to('cuda')

    while True:
        ret, frame = cap.read()

        results = model(frame)[0]

        detections = sv.Detections.from_ultralytics(results)

        bounding_box_annotator = sv.MaskAnnotator(
            color_lookup=ColorLookup.INDEX)
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
