from collections import defaultdict
import numpy as np
from ultralytics import YOLO
import cv2
import torch

class Regular_Line:
    #### 画规则线
    def __init__(self, a_x, a_y, b_x, b_y, mode=0):
        self.a_x = a_x
        self.a_y = a_y
        self.b_x = b_x
        self.b_y = b_y
        self.mode = mode  # mode表示方向，点a在左上方，因此mode=0表示从下往上是进；mode=1表示从上往下是进
        self.k_x = 0
        self.k_y = 0
        k = (a_y - b_y) / (a_x - b_x)
        if mode == 0:
            if a_y == b_y:
                self.k_x = 0
                self.k_y = 1
            elif a_x == b_x:
                self.k_x = 1
                self.k_y = 0
            else:
                self.k_x = 1
                self.k_y = -1 / k
        elif mode == 1:
            if a_y == b_y:
                self.k_x = 0
                self.k_y = -1
            elif a_x == b_x:
                self.k_x = -1
                self.k_y = 0
            else:
                self.k_x = 1
                self.k_y = 1 / k


def crossProduct(A_x, A_y ,B_x ,B_y, C_x, C_y):
    return (B_x - A_x) * (C_y - A_y) - (B_y - A_y) * (C_x - A_x);


###用于计算人是否跨线
def isLineIntersect(line, old_x, old_y, cur_x, cur_y):
    cp1 = crossProduct(line.a_x, line.a_y, line.b_x, line.b_y, old_x, old_y)
    cp2 = crossProduct(line.a_x, line.a_y, line.b_x, line.b_y, cur_x, cur_y)

    cp3 = crossProduct(old_x, old_y, cur_x, cur_y, line.a_x, line.a_y)
    cp4 = crossProduct(old_x, old_y, cur_x, cur_y, line.b_x, line.b_y)
    # positive and negative symbols
    if ((cp1 * cp2 <= 0) and (cp3 * cp4 <= 0)):
        return True
    return False


if __name__ == '__main__':
    # Load a model
    model = YOLO("yolov8_person_head.pt")  # load a pretrained model
    # model.model = model.model.cuda()  ### 如果跑起来很慢，可以把这一行打开，用GPU跑
    line = Regular_Line(144, 508, 2078, 471, 1)  ## 设置规则线

    entry = 0
    leave = 0
    total = 0
    font_scale = 1
    font_color = (255, 255, 0)
    line_type = 2
    # Open the video file
    video_path = "431710838.mp4"
    cap = cv2.VideoCapture(video_path)
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", show=False)
            total = torch.count_nonzero(results[0].boxes.cls)
            for track in model.predictor.trackers[0].tracked_stracks:
                # 如果不是人头，或者tracker不是活跃的
                if track.cls != 0.0 or not track.is_activated:
                    continue
                old_x = track.old_x
                old_y = track.old_y

                cur_x = track.tlwh[0] + 0.5 * track.tlwh[2]
                cur_y = track.tlwh[1] + 0.5 * track.tlwh[3]
                tmp_x = cur_x - old_x
                tmp_y = cur_y - old_y
                if isLineIntersect(line, old_x, old_y, cur_x, cur_y) and track.gap == 0:
                    track.gap = 30
                    if tmp_x * line.k_x + tmp_y * line.k_y > 0 and not track.is_entry:
                        entry += 1
                        track.is_entry = True
                    elif tmp_x * line.k_x + tmp_y * line.k_y < 0 and not track.is_leave:
                        leave += 1
                        track.is_leave = True

                    track.gap = max(track.gap - 1, 0)
                track.old_x = cur_x
                track.old_y = cur_y
            # Visualize the results on the frame
            # annotated_frame = results[0].plot()
            annotated_frame = frame
            cv2.putText(annotated_frame, f'total: {total}', (200, 100), 2, font_scale, font_color, line_type)

            cv2.putText(annotated_frame, f'entry: {entry}', (200, 200), 2, font_scale, font_color, line_type)
            cv2.putText(annotated_frame, f'leave: {leave}', (200, 300), 2, font_scale, font_color, line_type)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
            #
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()