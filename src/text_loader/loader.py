import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

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
        return self.vectorizer.fit_transform(tweets).toarray()

    def label_encoder(self, parties):
        self.encoder = LabelEncoder()
        return self.encoder.fit_transform(parties)

    def preprocess_tweets(self):
        self.data.Tweet = self.data.Tweet.apply(self.clean_text)
        return self.vectorize_text(self.data.Tweet.values)

    def preprocess_parties(self):
        self.data.Party = self.data.Party.apply(self.clean_text)
        return self.label_encoder(self.data.Party.values)
    
    def train_model(self):
        X = self.preprocess_tweets()
        y = self.preprocess_parties()
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X, y)
        return model
    
    def predict(self, model, new_tweets: list[str]):
        if self.vectorizer is None or self.encoder is None:
            raise ValueError("Model and encoder must be trained before prediction.")
        
        cleaned_tweets = [self.clean_text(tweet) for tweet in new_tweets]
        vectorized_tweets = self.vectorizer.transform(cleaned_tweets).toarray()
        predictions = model.predict(vectorized_tweets)
        self.model = model
        return self.encoder.inverse_transform(predictions)

