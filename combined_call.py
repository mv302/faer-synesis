import cv2
import numpy as np
from PIL import Image
import os
import time
from image_api_call import classify_image_call
from audio_api_call import classify_audio_call
import sounddevice as sd
from scipy.io.wavfile import write
import csv
import uuid
from datetime import datetime

def capture_video_frame(output_file="captured_frame.jpg"):
    """Capture a single frame from the webcam and save it as a JPG file."""
    print("Capturing video frame...")
    
    cap = cv2.VideoCapture(0)  # Open the default camera
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    
    # Give the camera a moment to adjust
    time.sleep(0.5)
    
    # Capture a single frame
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Convert the frame to a PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Save the image as JPG
        image.save(output_file, format="JPEG")
        print(f"Image saved to {output_file}")
        return output_file
    
    print("Failed to capture video frame")
    return None

def classify_image(image_file):
    print(f"Classifying image: {image_file}")
    # Example of how you might implement this with your API
    with open(image_file, 'rb') as f:
        image_data = f.read()
        return classify_image_call(image_data)
    #return "Image classification result would appear here"

def capture_audio():
    print("Capturing audio...")
    duration = 5  # seconds
    sample_rate = 44100  # Hz

    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype='int16')
    sd.wait()  # Wait until recording is finished

    write("captured_audio.wav", sample_rate, audio_data)
    audio_file = "captured_audio.wav"
        
    print(f"Audio captured and saved to {audio_file}")
    return audio_file

def main():    
    # Initialize CSV file
    csv_file = "classification_results.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["session_id", "timestamp", "image_result", "audio_result"])

    session_id = str(uuid.uuid4())  # Generate a unique session ID
    start_time = datetime.now()

    while True:
        # Capture video frame
        image_file = capture_video_frame()
        audio_file = capture_audio()
        timestamp = (datetime.now() - start_time).total_seconds()  # Calculate timestamp

        image_result = None
        audio_result = None

        if image_file:
            image_result = classify_image(image_file)
            print(f"\nImage Classification Result: {image_result}")
        if audio_file:
            audio_result = classify_audio_call(audio_file)
            print(f"\nAudio Classification Result: {audio_result}")

        # Store results in CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([session_id, timestamp, image_result, audio_result])
        
if __name__ == "__main__":
    main()