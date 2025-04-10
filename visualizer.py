import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import matplotlib.dates as mdates
from datetime import datetime, timedelta

def load_data(file_path):
    """Load emotion data from CSV file."""
    df = pd.read_csv(file_path)
    # Convert timestamp to seconds
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    return df

def visualize_emotion_distribution(df):
    """Create bar charts showing the distribution of emotions."""
    plt.figure(figsize=(15, 10))
    
    # Image emotions distribution
    plt.subplot(2, 1, 1)
    sns.countplot(x='image_result', data=df, palette='viridis')
    plt.title('Distribution of Image Emotions', fontsize=16)
    plt.xlabel('Emotion', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Audio emotions distribution
    plt.subplot(2, 1, 2)
    sns.countplot(x='audio_result', data=df, palette='magma')
    plt.title('Distribution of Audio Emotions', fontsize=16)
    plt.xlabel('Emotion', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('emotion_distribution.png', dpi=300)
    plt.close()
    print("Saved emotion distribution chart as 'emotion_distribution.png'")

def visualize_emotion_correlation(df):
    """Create a heatmap showing correlation between image and audio emotions."""
    # Create a cross-tabulation of image and audio emotions
    emotion_cross = pd.crosstab(df['image_result'], df['audio_result'], normalize='index')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(emotion_cross, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Correlation Between Image and Audio Emotions', fontsize=16)
    plt.xlabel('Audio Emotion', fontsize=14)
    plt.ylabel('Image Emotion', fontsize=14)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12, rotation=0)
    
    plt.tight_layout()
    plt.savefig('emotion_correlation.png', dpi=300)
    plt.close()
    print("Saved emotion correlation heatmap as 'emotion_correlation.png'")

def visualize_session_timeline(df, num_sessions=5):
    """Create timeline visualizations for a few sample sessions."""
    # Get unique session IDs
    sessions = df['session_id'].unique()
    
    # Sample a few sessions to visualize
    sample_sessions = np.random.choice(sessions, min(num_sessions, len(sessions)), replace=False)
    
    for session_id in sample_sessions:
        # Filter data for this session
        session_data = df[df['session_id'] == session_id].sort_values('timestamp')
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Create a base date for visualization (arbitrary, just for display)
        base_date = datetime(2023, 1, 1)
        
        # Convert timestamps to datetime for better x-axis formatting
        time_points = [base_date + timedelta(seconds=float(t)) for t in session_data['timestamp']]
        
        # Get unique emotions for consistent coloring
        unique_image_emotions = df['image_result'].unique()
        unique_audio_emotions = df['audio_result'].unique()
        
        # Create color maps
        image_colors = sns.color_palette("viridis", len(unique_image_emotions))
        audio_colors = sns.color_palette("magma", len(unique_audio_emotions))
        
        image_color_map = dict(zip(unique_image_emotions, image_colors))
        audio_color_map = dict(zip(unique_audio_emotions, audio_colors))
        
        # Plot image emotions
        last_time = time_points[0]
        last_emotion = session_data['image_result'].iloc[0]
        
        for i in range(len(time_points)):
            current_time = time_points[i]
            current_emotion = session_data['image_result'].iloc[i]
            
            # Draw rectangle for the emotion duration
            ax1.axvspan(last_time, current_time, 
                       color=image_color_map[last_emotion], 
                       alpha=0.7)
            
            # Update for next segment
            last_time = current_time
            last_emotion = current_emotion
        
        # Plot audio emotions
        last_time = time_points[0]
        last_emotion = session_data['audio_result'].iloc[0]
        
        for i in range(len(time_points)):
            current_time = time_points[i]
            current_emotion = session_data['audio_result'].iloc[i]
            
            # Draw rectangle for the emotion duration
            ax2.axvspan(last_time, current_time, 
                       color=audio_color_map[last_emotion], 
                       alpha=0.7)
            
            # Update for next segment
            last_time = current_time
            last_emotion = current_emotion
        
        # Add data points
        for i, time_point in enumerate(time_points):
            ax1.plot(time_point, 0.5, 'ko', markersize=8)
            ax2.plot(time_point, 0.5, 'ko', markersize=8)
        
        # Add legend for image emotions
        image_patches = [plt.Rectangle((0, 0), 1, 1, color=image_color_map[emotion]) 
                         for emotion in unique_image_emotions]
        ax1.legend(image_patches, unique_image_emotions, loc='upper right')
        
        # Add legend for audio emotions
        audio_patches = [plt.Rectangle((0, 0), 1, 1, color=audio_color_map[emotion]) 
                         for emotion in unique_audio_emotions]
        ax2.legend(audio_patches, unique_audio_emotions, loc='upper right')
        
        # Format x-axis to show time
        formatter = mdates.DateFormatter('%M:%S')  # Minutes:Seconds
        ax1.xaxis.set_major_formatter(formatter)
        ax2.xaxis.set_major_formatter(formatter)
        
        # Set labels and titles
        ax1.set_title(f'Image Emotions - Session {session_id[:8]}...', fontsize=16)
        ax1.set_ylabel('Emotion State', fontsize=14)
        ax1.set_yticks([])
        
        ax2.set_title(f'Audio Emotions - Session {session_id[:8]}...', fontsize=16)
        ax2.set_xlabel('Time (MM:SS)', fontsize=14)
        ax2.set_ylabel('Emotion State', fontsize=14)
        ax2.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(f'session_timeline_{session_id[:8]}.png', dpi=300)
        plt.close()
        print(f"Saved timeline for session {session_id[:8]}... as 'session_timeline_{session_id[:8]}.png'")

def visualize_emotion_transitions(df):
    """Create transition matrices showing how emotions change over time."""
    plt.figure(figsize=(20, 10))
    
    # Process image emotion transitions
    image_transitions = defaultdict(lambda: defaultdict(int))
    audio_transitions = defaultdict(lambda: defaultdict(int))
    
    # Group by session to track transitions within each session
    for session_id in df['session_id'].unique():
        session_data = df[df['session_id'] == session_id].sort_values('timestamp')
        
        # Process image emotions
        prev_image = None
        for emotion in session_data['image_result']:
            if prev_image is not None:
                image_transitions[prev_image][emotion] += 1
            prev_image = emotion
        
        # Process audio emotions
        prev_audio = None
        for emotion in session_data['audio_result']:
            if prev_audio is not None:
                audio_transitions[prev_audio][emotion] += 1
            prev_audio = emotion
    
    # Convert to DataFrames for visualization
    image_emotions = sorted(df['image_result'].unique())
    audio_emotions = sorted(df['audio_result'].unique())
    
    image_matrix = pd.DataFrame(0, index=image_emotions, columns=image_emotions)
    audio_matrix = pd.DataFrame(0, index=audio_emotions, columns=audio_emotions)
    
    for from_emotion, transitions in image_transitions.items():
        for to_emotion, count in transitions.items():
            image_matrix.at[from_emotion, to_emotion] = count
    
    for from_emotion, transitions in audio_transitions.items():
        for to_emotion, count in transitions.items():
            audio_matrix.at[from_emotion, to_emotion] = count
    
    # Normalize by row (from each starting emotion)
    image_matrix = image_matrix.div(image_matrix.sum(axis=1), axis=0).fillna(0)
    audio_matrix = audio_matrix.div(audio_matrix.sum(axis=1), axis=0).fillna(0)
    
    # Plot transitions
    plt.subplot(1, 2, 1)
    sns.heatmap(image_matrix, annot=True, cmap='Blues', fmt='.2f', linewidths=.5)
    plt.title('Image Emotion Transitions', fontsize=16)
    plt.xlabel('To Emotion', fontsize=14)
    plt.ylabel('From Emotion', fontsize=14)
    
    plt.subplot(1, 2, 2)
    sns.heatmap(audio_matrix, annot=True, cmap='Reds', fmt='.2f', linewidths=.5)
    plt.title('Audio Emotion Transitions', fontsize=16)
    plt.xlabel('To Emotion', fontsize=14)
    plt.ylabel('From Emotion', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('emotion_transitions.png', dpi=300)
    plt.close()
    print("Saved emotion transitions as 'emotion_transitions.png'")

def visualize_session_statistics(df):
    """Visualize statistics about session length and number of data points."""
    # Calculate session statistics
    session_stats = df.groupby('session_id').agg(
        max_timestamp=('timestamp', 'max'),
        num_points=('timestamp', 'count')
    ).reset_index()
    
    session_stats['session_duration_min'] = session_stats['max_timestamp'] / 60  # Convert to minutes
    
    plt.figure(figsize=(15, 10))
    
    # Histogram of session durations
    plt.subplot(2, 1, 1)
    sns.histplot(session_stats['session_duration_min'], bins=20, kde=True)
    plt.title('Distribution of Session Durations', fontsize=16)
    plt.xlabel('Session Duration (minutes)', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(linestyle='--', alpha=0.7)
    
    # Histogram of data points per session
    plt.subplot(2, 1, 2)
    sns.histplot(session_stats['num_points'], bins=15, kde=True)
    plt.title('Distribution of Data Points per Session', fontsize=16)
    plt.xlabel('Number of Data Points', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('session_statistics.png', dpi=300)
    plt.close()
    print("Saved session statistics as 'session_statistics.png'")

def main():
    # Replace with your CSV file path
    file_path = 'result_populated.csv'
    
    try:
        # Load data
        df = load_data(file_path)
        print(f"Loaded {len(df)} data points from {len(df['session_id'].unique())} sessions")
        
        # Create visualizations
        visualize_emotion_distribution(df)
        visualize_emotion_correlation(df)
        visualize_session_timeline(df, num_sessions=3)
        visualize_emotion_transitions(df)
        visualize_session_statistics(df)
        
        print("\nAll visualizations have been created successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()