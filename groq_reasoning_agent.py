import streamlit as st
import pandas as pd
from groq import Groq

# Initialize the Groq client
with open("groq_api.txt", "r") as file:
    api_key = file.read().strip()

client = Groq(
    api_key=api_key,
)

# Streamlit app
st.title("Emotion Analysis with Groq")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(data.head())

    # Sample the dataset to reduce token size
    sampled_data = data.sample(frac=0.25, random_state=42)
    data_string = sampled_data.to_csv(index=False)

    # Ensure the token count is less than 6000
    # if len(data_string.split()) > 6000:
    #     while len(data_string.split()) > 6000:
    #         data_string = sampled_data.to_csv(index=False)
    #         if len(data_string.split()) <= 6000:
    #             break
    # else:
        # Update the chat completion request
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": (
                    "You are given a dataset with session_id, timestamp, image_result, audio_result. "
                    "This is the emotion classified using images and audio of a particular child. "
                    "You have to analyze each session separately. Each session will last no longer than 45 minutes. "
                    "Give inference of the child's emotion and a comprehensive emotional profile for this child aesthetically, "
                    "so that a doctor can use it. The data starts from here:\n" + data_string
                ),
            }
        ],
        model="deepseek-r1-distill-qwen-32b",
        stream=False,
    )

    # Display the result
    st.subheader("Emotion Analysis Result:")
    st.write(chat_completion.choices[0].message.content)