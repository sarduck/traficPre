import cv2
import torch
import pandas as pd
from datetime import datetime, timedelta

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def process_video(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    print(frame_rate)

    start_time = datetime.strptime("04/03/2016 0:00", "%d/%m/%Y %H:%M")
    interval = timedelta(seconds=1)

    current_interval_start = start_time
    vehicle_count = 0

    records = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        results = model(frame)

        for i, detection in enumerate(results.xyxy[0]):
            x_min, y_min, x_max, y_max, confidence, class_id = detection
            class_id = int(class_id)
            class_name = model.names[class_id]

            if class_name in ["car", "truck", "bus", "motorcycle"]:
                    vehicle_count += 1

        # Calculate the current time in the video
        current_time = start_time + timedelta(seconds=(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000))

        if current_time >= current_interval_start + interval:
            records.append({
                "per sec": current_interval_start.strftime("%d/%m/%Y %H:%M:%S"),
                "Lane 1 Flow (Veh/sec)": round(vehicle_count/frame_rate),
            })
            # Reset for next interval
            current_interval_start += interval
            vehicle_count = 0

    cap.release()

    # Save results to CSV in the required format
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved: {output_csv}")

#adjust path accordingly
video_path = "vid6.mp4"  
output_csv = "data6.csv"  

process_video(video_path, output_csv)
