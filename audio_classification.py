# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("audio-classification", model="firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3")

# Load model directly
from transformers import AutoProcessor, AutoModelForAudioClassification

processor = AutoProcessor.from_pretrained("firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3")
model = AutoModelForAudioClassification.from_pretrained("firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3")

def classify_audio(audiofile):
    emotion = pipe(audiofile, sampling_rate=16000)
    print(emotion)
    return emotion

# from huggingface_hub import InferenceClient

# client = InferenceClient(
#     provider="hf-inference",
#     api_key="hf_xxxxxxxxxxxxxxxxxxxxxxxx",
# )

# output = client.audio_classification("sample1.flac", model="firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3")