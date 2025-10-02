import string
import re
import os
import mlflow.xgboost
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

ML_RUNS_DIR = "/usr/mlruns"
MODEL_PATH = os.path.join(ML_RUNS_DIR, "595336408377215950/models/m-ce7cb0cea5ab416e926ed6587a523050/artifacts")
ENCODER_PATH = os.path.join(ML_RUNS_DIR, "595336408377215950/models/m-484f58f86126427e9ee592b7017686be/artifacts")
VECTORIZER_PATH = os.path.join(ML_RUNS_DIR, "595336408377215950/models/m-03b98d12ac644c68804441ce9cd769c5/artifacts")

class DataLoader:
    def __init__(self, filepath="data/Tweets.csv"):
        self.filepath = filepath
        self.data = self.load_data()
        self.vectorizer = None
        self.encoder = None

    def load_data(self):
        """Loads data from a CSV file."""
        return pd.read_csv(self.filepath)

    @staticmethod
    def remove_characters(text: str) -> str:
        """Remove non-letters from a given string"""
        remove_chars = string.punctuation
        translator = str.maketrans('', '', remove_chars)
        text = re.sub(r'\d+', '', text)

        return text.translate(translator)

    def clean_text(self, text) -> str:
        """Keep only retain words in a given string"""
        text = self.remove_characters(str(text))
        return text.rsplit(' ', 1)[0]

    def vectorize_text(self, tweets: list[str]):
        self.vectorizer = TfidfVectorizer(max_features=2500, min_df=1, max_df=0.8)
        return self.vectorizer.fit_transform(tweets).toarray() # type: ignore

    def label_encoder(self, parties):
        self.encoder = LabelEncoder()
        return self.encoder.fit_transform(parties)

    def preprocess_tweets(self):
        self.data.Tweet = self.data.Tweet.apply(self.clean_text)
        return self.vectorize_text(self.data.Tweet.values) # type: ignore

    def preprocess_parties(self):
        self.data.Party = self.data.Party.apply(self.clean_text)
        return self.label_encoder(self.data.Party.values)
    
    def predict(self, new_tweets: list[str]):
        self.vectorizer = mlflow.sklearn.load_model(VECTORIZER_PATH)
        self.encoder = mlflow.sklearn.load_model(ENCODER_PATH)
        self.model = mlflow.xgboost.load_model(MODEL_PATH)

        cleaned_tweets = [self.clean_text(tweet) for tweet in new_tweets]
        vectorized_tweets = self.vectorizer.transform(cleaned_tweets).toarray() # type: ignore
        predictions = self.model.predict(vectorized_tweets)
        
        return self.encoder.inverse_transform(predictions) # type: ignore

