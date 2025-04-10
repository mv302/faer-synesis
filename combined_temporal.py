import pyaudio
import wave
import time
import cv2
import streamlit as st
from audio_classification import classify_audio
from image_classification import classify_image  # Assume this is your video classification module

# Audio recording settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 7
AUDIO_OUTPUT_FILENAME = "recorded_audio.wav"
VIDEO_OUTPUT_FILENAME = "recorded_video.avi"

def record_audio():
    audio = pyaudio.PyAudio()

    # Open the audio stream
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    st.write("Recording audio...")

    frames = []

    # Record for the specified duration
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    st.write("Audio recording complete.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio to a file
    with wave.open(AUDIO_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    return AUDIO_OUTPUT_FILENAME

def record_video():
    st.write("Recording video...")
    cap = cv2.VideoCapture(0)  # Open the webcam
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(VIDEO_OUTPUT_FILENAME, fourcc, 20.0, (640, 480))

    start_time = time.time()
    while int(time.time() - start_time) < RECORD_SECONDS:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            # Display the video frame in Streamlit
            st.image(frame, channels="BGR")
        else:
            break

    cap.release()
    out.release()
    st.write("Video recording complete.")

    return VIDEO_OUTPUT_FILENAME

def main():
    st.title("Audio and Video Emotion Classification")

    if st.button("Start Recording"):
        # Record audio
        audio_file = record_audio()

        # Record video
        video_file = record_video()

        # Perform emotion classification on the recorded audio
        audio_emotion = classify_audio(audio_file)
        st.write(f"Audio Emotion: {audio_emotion}")

        # Perform emotion classification on the recorded video
        video_emotion = classify_image(video_file)
        st.write(f"Video Emotion: {video_emotion}")

if __name__ == "__main__":
    main()
