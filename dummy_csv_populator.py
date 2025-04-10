import csv
import uuid
import random
from datetime import datetime, timedelta

# Emotions that can be detected
image_emotions = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]
audio_emotions = ["happy", "sad", "angry", "fearful", "surprised", "neutral", "calm"]

# Generate 1000 rows of data across multiple sessions
def generate_emotion_session_data(num_rows=1000):
    data = []
    rows_generated = 0
    
    while rows_generated < num_rows:
        # Create a new session with a unique ID
        session_id = str(uuid.uuid4())
        
        # Random session length between 15 and 45 minutes (in seconds)
        session_length = random.randint(15 * 60, 45 * 60)
        
        # Number of data points in this session (between 5 and 15)
        num_points = random.randint(5, 15)
        
        # Generate timestamps for this session
        timestamps = sorted([random.randint(1, session_length) for _ in range(num_points)])
        
        # Create "emotion states" that tend to persist for a while
        current_image_emotion = random.choice(image_emotions)
        current_audio_emotion = random.choice(audio_emotions)
        
        for timestamp in timestamps:
            # 30% chance to change the emotion at each timestamp
            if random.random() < 0.3:
                current_image_emotion = random.choice(image_emotions)
            if random.random() < 0.3:
                current_audio_emotion = random.choice(audio_emotions)
                
            # Add the data point
            data.append({
                'session_id': session_id,
                'timestamp': timestamp,
                'image_result': current_image_emotion,
                'audio_result': current_audio_emotion
            })
            
            rows_generated += 1
            if rows_generated >= num_rows:
                break

    return data

# Generate 1000 rows of data
emotion_data = generate_emotion_session_data(1000)

# Write to CSV string
csv_file_path = "result_populated.csv"
with open(csv_file_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["session_id", "timestamp", "image_result", "audio_result"])
    for row in emotion_data:
        writer.writerow([row['session_id'], f"{row['timestamp']:.6f}", row['image_result'], row['audio_result']])

