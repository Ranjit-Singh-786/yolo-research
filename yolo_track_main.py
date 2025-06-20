from ultralytics import YOLO
from live_detect import idintifie_person, find_top_point_bbox, round_values
import cv2 as cv
import numpy as np
from Tracking_id import SimpleTracker

tracker = SimpleTracker(iou_threshold=0.4)

video_path = r"testing video\second.mp4"   # video resolution "1440, 2560, 3"
model = YOLO("yolov8n.pt")

cap = cv.VideoCapture(video_path)
total_known_person_names = set()
total_unknown_person_names = set()
person_count = 0

past_person_frames_memory = []  # desired format will be stored here
past_face_frames_memory = []
frame_no = 1
frame_count_face = 1

while cap.isOpened():
    ret, frame = cap.read()
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.cvtColor(gray_frame, cv.COLOR_GRAY2BGR)      ##  # convert back to 3-channels for drawing
    if not ret:
        break

    # frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
    frame = cv.resize(frame, (2000, 1200))   # 1200,1100
    height, width, _ = frame.shape
    mid_x = width // 2
    blue_line_x_cord = mid_x - 300
    red_line_x_cord = mid_x + 150

    frame, known_person_names, unknown_person_names, detected_face_bbox_in_region, boxes = idintifie_person(
        frame, red_line_x_cord, blue_line_x_cord)

    total_known_person_names.update(known_person_names)
    total_unknown_person_names.update(unknown_person_names)

    results = model.track(frame, conf=0.80, classes=[0])
    result = results[0]
    bbox_results = round_values(result.boxes.xyxy.tolist())

    frame_objects = []
    for det in range(len(result)):
        box_sample = dict()
        box_id = int(result[det].boxes.id.tolist()[0])
        box_cord = result[det].boxes.xyxy.tolist()
        box_cord  = list(map(int, box_cord[0]))
        x1, y1 , x2 , y2 = box_cord 
        TL = (x1,y1) 
        TR = (x2,y1)     
        top_center_x = (TL[0] + TR[0]) // 2
        top_center_y = TL[1]   
        
        cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv.circle(frame, (top_center_x, top_center_y), 5, (0, 255, 0), -1)
        cv.putText(frame, f"ID {box_id}", (x1, y1 - 10),
            cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if blue_line_x_cord < top_center_x < red_line_x_cord:   # is box in region 
            box_sample["ID"] = box_id 
            box_sample['BBOX'] = box_cord 
            frame_objects.append(box_sample) 

    if frame_objects:
        if (len(past_person_frames_memory) < 10) :
            past_person_frames_memory.append({f"frame_{frame_no}": frame_objects})
            frame_no += 1
        else:
            print("Your Detected Person Memory:\n", past_person_frames_memory)
            past_person_frames_memory.clear()
            frame_no = 1
            past_person_frames_memory.append({f"frame_{frame_no}": frame_objects})
            frame_no += 1

    # Face memory storage (unchanged)
    if (len(past_face_frames_memory) < 10) and (len(detected_face_bbox_in_region) >= 1):
        past_face_frames_memory.append({f"frame_{frame_count_face}": detected_face_bbox_in_region})
        frame_count_face += 1
    else:
        if (len(past_face_frames_memory) == 10):
            print("Your Detected Person Face Memory:\n", past_face_frames_memory)
            past_face_frames_memory.clear()
            frame_count_face = 1

        if (len(detected_face_bbox_in_region) >= 1):
            past_face_frames_memory.append({f"frame_{frame_count_face}": detected_face_bbox_in_region})
            frame_count_face += 1

    cv.line(frame, (blue_line_x_cord, 0), (blue_line_x_cord, height), color=(0, 255, 0), thickness=2)
    cv.line(frame, (red_line_x_cord, 0), (red_line_x_cord, height), color=(0, 0, 255), thickness=2)

    cv.putText(frame, f"Person IN : {person_count}", (20, 50),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4)
    cv.imshow("Annotated TEST Video", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
