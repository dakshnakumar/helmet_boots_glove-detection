# from roboflow import Roboflow
# rf = Roboflow(api_key="xiWkPMNK66UFv4chCfGS")
# project = rf.workspace().project("helmet-iw8mn")
# model = project.version(2).model

# # infer on a local image
# print(model.predict("/Volumes/Project/Ai project/deep learning/helmet detection/handsome-indianasian-man-helmet-over-260nw-1181201782.webp", confidence=40, overlap=30).json())

# # visualize your prediction
# # model.predict("/Volumes/Project/Ai project/deep learning/helmet detection/handsome-indianasian-man-helmet-over-260nw-1181201782.webp", confidence=40, overlap=30).save("prediction.jpg")

# # infer on an image hosted elsewhere
# # print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())


import cv2
import argparse
import supervision as sv
from ultralytics import YOLO
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


ZONE_POLYGON = np.array([
    [0,0],
    [1280,0],
    [1250,720],
    [0,720]
])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOV8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280,720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width,frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)

    model = YOLO("/Volumes/Project/external project/helmet_boot_glove detection/src/model/best.pt")
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
    # zone = sv.PolygonZone()
    while True:
        ret , frame = cap.read()

        result = model(frame)[0]
        detection = sv.Detections.from_ultralytics(result)
        detection = tracker.update_with_detections(detection)
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _,confidence,class_id,_
            in zip(detection.tracker_id, detection.confidence, detection.class_id, detection.xyxy)
        ]
        frame = box_annotator.annotate(scene=frame,detections=detection)
        cv2.imshow('yolov8',frame)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

if __name__ == "__main__":
    main()