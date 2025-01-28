#atcc final
import cv2
import numpy as np
from ultralytics import YOLO

def load_videos_from_folder(videos_folder):
    """Load all video files from the provided folder."""
    import os
    video_files = []
    for filename in os.listdir(videos_folder):
        if filename.endswith(".mp4"):
            video_files.append({"path": os.path.join(videos_folder, filename), "road_name": filename})
    return video_files

def process_frame(frame, model, road_name, directions):
    """
    Process a video frame: Detect objects, classify directions, and overlay information.
    Args:
        frame: The input video frame.
        model: The YOLO model for object detection.
        road_name: Name of the road the video corresponds to.
        directions: Dictionary to count vehicle directions.
    Returns:
        Processed frame with detection overlays and vehicle count per direction.
    """
    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()
    vehicle_count = {"car": 0, "truck": 0, "motorcycle": 0, "bus": 0}
    
    for det in detections:
        x1, y1, x2, y2, conf, class_id = det
        class_id = int(class_id)
        label = model.names[class_id]
        
        if label in vehicle_count:
            vehicle_count[label] += 1
            center_x = (x1 + x2) / 2
            if center_x < frame.shape[1] / 2:
                directions['left'] += 1
            else:
                directions['right'] += 1

            color = (0, 255, 0) if conf > 0.5 else (0, 0, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, f"Road: {road_name}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    y_offset = 50
    for direction, count in directions.items():
        cv2.putText(frame, f"{direction.capitalize()}: {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 25

    total_vehicles = sum(vehicle_count.values())
    cv2.putText(frame, f"Total Vehicles: {total_vehicles}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 25

    # Determine signal based on vehicle count
    signal_color, signal_rgb = determine_signal(total_vehicles)
    cv2.putText(frame, f"Signal: {signal_color}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, signal_rgb, 2)

    return frame, vehicle_count

def determine_signal(total_vehicles):
    """Determine the signal color based on vehicle count."""
    if total_vehicles < 10:
        return "Green", (0, 255, 0)
    elif total_vehicles < 20:
        return "Yellow", (0, 255, 255)
    else:
        return "Red", (0, 0, 255)

def process_videos(videos_folder, model):
    """Process videos from the folder and display results."""
    videos = load_videos_from_folder(videos_folder)
    caps = []
    road_names = []
    
    for video in videos:
        cap = cv2.VideoCapture(video["path"])
        if not cap.isOpened():
            print(f"Error: Could not open video stream for {video['road_name']}.")
            continue
        caps.append(cap)
        road_names.append(video["road_name"])

    # Define the target size for all frames (ensure consistent dimensions for stacking)
    target_width, target_height = 640, 480

    while True:
        processed_frames = []
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                print(f"End of video stream for {road_names[i]}.")
                caps[i].release()
                continue

            directions = {"left": 0, "right": 0}
            frame_resized = cv2.resize(frame, (target_width, target_height))  # Resize all frames to the same size

            # Process frame and add all necessary overlays, including the signal
            processed_frame, vehicle_count = process_frame(frame_resized, model, road_names[i], directions)

            processed_frames.append(processed_frame)

        if len(processed_frames) == 0:
            break

        # Stack the frames into a grid
        rows = []
        for i in range(0, len(processed_frames), 2):
            row = np.hstack(processed_frames[i:i + 2]) if i + 1 < len(processed_frames) else processed_frames[i]
            rows.append(row)

        # Ensure all rows have consistent height for vertical stacking
        row_heights = [row.shape[0] for row in rows]
        max_height = max(row_heights)
        rows_resized = [cv2.resize(row, (target_width, max_height)) for row in rows]

        grid_frame = np.vstack(rows_resized)  # Stack the rows into a grid

        cv2.imshow("Traffic Management System", grid_frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    videos_folder = r"C:\Users\Mugdhi Saxena\Documents\Mugdhi S\Infosys Springboard\internship\project files\videos"  
    model = YOLO("yolov8n.pt")
    process_videos(videos_folder, model)
