import streamlit as st
import requests


def get_prediction(input_text):
    r = requests.post(
        "http://model_inference_endpoint:8000/get-prediction",
        json={"input_texts": input_text},
    ).json()
    return r["prediction"]


# Streamlit page configuration
st.set_page_config(page_title="Tweet Classifier", layout="wide")

# Streamlit UI components
st.title("Classify your tweet")

# User inputs the tweet
tweet_input = st.text_input("Enter your tweet", "")

# Button to trigger prediction
if st.button("Classify Tweet"):
    # Get prediction
    prediction = get_prediction(tweet_input)
    st.write(
        "Prediction: ",
        (
            f"**:blue[{prediction}]**"
            if prediction == "Democrat"
            else f"**:red[{prediction}]**"
        ),
    )
