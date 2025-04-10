from huggingface_hub import InferenceClient

with open("huggingface_token.txt", "r") as token_file:
    api_key = token_file.read().strip()

client = InferenceClient(
    provider="hf-inference",
    api_key=api_key,
)

def classify_audio_call(audiofile):
    try:
        output = client.audio_classification(audiofile, model="firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3")
        return output[0]['label']
    except Exception as e:
        return "failed"

