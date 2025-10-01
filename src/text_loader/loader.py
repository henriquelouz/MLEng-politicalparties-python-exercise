import pandas as pd
import string
import re
import mlflow.xgboost

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = 'file:///Users/henriquelouzada/repos/MLEng-politicalparties-python-exercise/mlruns/595336408377215950/models/m-359665d13f1345aa8d0181435c0fc9f6/artifacts'
# TODO: change to relative path

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
        if self.vectorizer is None or self.encoder is None:
            raise ValueError("Model and encoder must be trained before prediction.")

        # TODO: save/load vectorizer and encoder from mlflow
        
        model = mlflow.xgboost.load_model(MODEL_PATH)

        cleaned_tweets = [self.clean_text(tweet) for tweet in new_tweets]
        vectorized_tweets = self.vectorizer.transform(cleaned_tweets).toarray() # type: ignore
        predictions = model.predict(vectorized_tweets)
        self.model = model
        return self.encoder.inverse_transform(predictions)

