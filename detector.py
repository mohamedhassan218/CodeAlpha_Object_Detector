"""
    Module handles the process of choosing the version of the YOLO model,
    and track the objects with it in case of using the webcam or choose
    a specific video to work with.
    
    @author  Mohamed Hassan
    @since   2024-5-2
"""

import cv2
from ultralytics import YOLO
from enum import Enum
import numpy as np
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict


class YOLOType(Enum):
    """
    Enumeration for supported YOLO model sizes.

    This class defines the different YOLO model variants that can be used
    with the video and webcam handlers. Each member corresponds to a specific
    pre-trained model file name.

    [nano, small, medium, large, x-large]
    """

    YOLOv8n = "yolov8n.pt"
    YOLOv8s = "yolov8s.pt"
    YOLOv8m = "yolov8m.pt"
    YOLOv8l = "yolov8l.pt"
    YOLOv8x = "yolov8x.pt"


def video_handler(filename: str):
    """
    Function that analyzes a video file using YOLO model, tracks detected
    objects, and saves the results in a new video file.

    @param:
        filename: string represents the path of the video.

    @return: None
    """

    model = YOLO(YOLOType.YOLOv8n.value)
    names = model.model.names

    # Dictionary for history object tracking.
    track_history = defaultdict(lambda: [])

    # Open the video file.
    cap = cv2.VideoCapture(filename)
    assert (
        cap.isOpened()
    ), "Error reading video file"  # Ensure that the file opened successfully.

    # Get video properties: width, height and frames per second.
    w, h, fps = (
        int(cap.get(x))
        for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
    )

    # Create file to save the output inside: result.avi
    result = cv2.VideoWriter("result.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Iterate through each frame of the video to read, detect and save into result.avi.
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.track(frame, persist=True, verbose=False)
            boxes = results[0].boxes.xyxy.cpu()
            if results[0].boxes.id is not None:
                clss = results[0].boxes.cls.cpu().tolist()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                confs = results[0].boxes.conf.float().cpu().tolist()
                annotator = Annotator(frame, line_width=2)
                for box, cls, track_id in zip(boxes, clss, track_ids):
                    annotator.box_label(
                        box, color=colors(int(cls), True), label=names[int(cls)]
                    )
                    track = track_history[track_id]
                    track.append(
                        (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
                    )
                    if len(track) > 30:
                        track.pop(0)
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.circle(frame, (track[-1]), 7, colors(int(cls), True), -1)
                    cv2.polylines(
                        frame,
                        [points],
                        isClosed=False,
                        color=colors(int(cls), True),
                        thickness=2,
                    )
            result.write(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                # When it reachs EOF.
                break
        else:
            break

    result.release()
    cap.release()
    cv2.destroyAllWindows()


def webcam_handler():
    """
    Function that handles the process of detecting live objects from the webcam.
    """

    model = YOLO(YOLOType.YOLOv8n.value)
    model.predict(source="0", show=True)


# Test our work.
if __name__ == "__main__":
    webcam_handler(YOLOType.YOLOv8n)
