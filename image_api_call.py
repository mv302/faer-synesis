from huggingface_hub import InferenceClient

with open("huggingface_token.txt", "r") as token_file:
    api_key = token_file.read().strip()

client = InferenceClient(
    provider="hf-inference",
    api_key=api_key,
)

def classify_image_call(imagefile):
    try:
        output = client.image_classification(imagefile, model="dima806/facial_emotions_image_detection")
        return output[0]['label']
    except Exception as e:
        return "failed"