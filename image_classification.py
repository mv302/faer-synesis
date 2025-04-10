# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")


def classify_image(imagefile):
    emotion = pipe(imagefile)
    print(emotion)
    return emotion

