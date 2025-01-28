#running on diff windows from folder
import cv2
import os
import numpy as np
from ultralytics import YOLO
import pytesseract
import threading

# Initialize YOLO v8 model
model = YOLO('yolov8n.pt')

# Function to process each video
def process_video(video_path, road_name, max_vehicle_limit):
    cap = cv2.VideoCapture(video_path)
    vehicle_count = 0
    traffic_signal = 'RED'
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get the original frame rate of the video
    frame_delay = int(1000 / fps)  # Delay between frames in milliseconds
    frame_count = 0  # Initialize frame count

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames for faster processing
        if frame_count % 2 != 0:  # Process every 2nd frame
            frame_count += 1
            continue

        # YOLO object detection
        results = model(frame)
        detections = results[0].boxes.data.cpu().numpy()  # Access bounding boxes
        vehicle_count = len(detections)  # Assuming all detected objects are vehicles

        # Update traffic signal dynamically
        if vehicle_count < max_vehicle_limit * 0.5:
            traffic_signal = 'GREEN'
        elif vehicle_count < max_vehicle_limit:
            traffic_signal = 'YELLOW'
        else:
            traffic_signal = 'RED'

        # Annotate frame
        for *box, conf, cls in detections:
            if len(box) == 4:  # Ensure box has valid coordinates
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'ID: {int(cls)}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # OCR for license plate recognition
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        license_plate_text = pytesseract.image_to_string(gray_frame, config='--psm 11')

        # Display video with annotations
        annotated_frame = frame.copy()
        cv2.putText(annotated_frame, f'{road_name}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f'Vehicle Count: {vehicle_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f'Traffic Light: {traffic_signal}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(f'{road_name}', annotated_frame)

        # Save results dynamically
        with open(f'{road_name}_results.txt', 'a') as f:
            f.write(f'Vehicle Count: {vehicle_count}, Traffic Light: {traffic_signal}, License Plate: {license_plate_text}\n')

        # Wait for the appropriate delay to match the video's original speed
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break

        frame_count += 1  # Increment frame count at the end of each loop

    cap.release()
    cv2.destroyAllWindows()

# Main function to process multiple videos in parallel
def main():
    video_folder = r'C:\Users\Mugdhi Saxena\Documents\Mugdhi S\Infosys Springboard\internship\project files\videos'  # Path to the folder containing videos
    max_vehicle_limit = 50  # Example maximum limit per volume

    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]

    # Parallel processing for videos
    threads = []
    for i, video_file in enumerate(video_files):
        road_name = f'Road-{i + 1}'
        video_path = os.path.join(video_folder, video_file)
        thread = threading.Thread(target=process_video, args=(video_path, road_name, max_vehicle_limit))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == '__main__':
    main()
