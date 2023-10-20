import argparse

from tqdm import tqdm
from ultralytics import YOLO

import supervision as sv


def process_video(
    source_weights_path: str,
    source_video_path: str,
    target_video_path: str,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
) -> None:
    print("processing video")
    model = YOLO(source_weights_path)
    print("model loaded")

    tracker = sv.ByteTrack()
    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    print("annotators loaded")
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path)
    print("frame generator loaded")
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    print("video info loaded")
    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            results = model(
                frame, verbose=False, conf=confidence_threshold, iou=iou_threshold
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)

            annotated_frame = box_annotator.annotate(
                scene=frame.copy(), detections=detections
            )

            annotated_labeled_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections
            )

            sink.write_frame(frame=annotated_labeled_frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video Processing with YOLO and ByteTrack"
    )
    # parser.add_argument(
    #     "--source_weights_path",
    #     required=True,
    #     help="Path to the source weights file",
    #     type=str,
    # )
    # parser.add_argument(
    #     "--source_video_path",
    #     required=True,
    #     help="Path to the source video file",
    #     type=str,
    # )
    # parser.add_argument(
    #     "--target_video_path",
    #     required=True,
    #     help="Path to the target video file (output)",
    #     type=str,
    # )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )

    args = parser.parse_args()

    process_video(
        "yolov8m.pt",  # source_weights_path=args.source_weights_path,
        "train.mp4",  # source_video_path=args.source_video_path,
        "tracking-train.mp4",  # target_video_path=args.target_video_path,
    )
