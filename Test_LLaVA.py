import cv2
import ollama
import time

# Replace with your actual video file name
video_path = '"C:\beach\data\raw\beach\fighting\crazy fight breaks off at the beach ｜ #fight #worldstar #fyp #shorts #beach [LiW9YUSt3Zg].mp4"'

# Open the video using OpenCV
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file '{video_path}'")
    exit()

# Get the frame rate (FPS) of the video
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video loaded. FPS: {fps}")

frame_count = 0
analyses_done = 0

print("Starting video analysis. Processing 1 frame per second (Max 3 frames for this test)...")

while cap.isOpened() and analyses_done < 3:
    ret, frame = cap.read()
    
    if not ret:
        print("End of video reached.")
        break
        
    # Process 1 frame every second (e.g., if FPS is 25, process frame 0, 25, 50)
    if frame_count % int(fps) == 0:
        print(f"\n--- Analyzing video at second {analyses_done} ---")
        
        # Convert the OpenCV frame (numpy array) into JPG image bytes for LLaVA
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()
        
        # Send the extracted frame to LLaVA
        response = ollama.generate(
            model='llava:13b',
            prompt='You are monitoring a beach safety camera. Briefly describe the scene. Is anyone fighting?',
            images=[image_bytes]
        )
        
        print(f"LLaVA: {response['response']}")
        analyses_done += 1
        
    frame_count += 1

# Clean up
cap.release()
print("\nVideo test complete!")