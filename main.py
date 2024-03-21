from string import punctuation
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os
from os.path import dirname, join, realpath
import joblib
import uvicorn
from fastapi import FastAPI 
import pandas as pd
from bertopic import BERTopic

# nltk.download('wordnet')

app = FastAPI(
    title="Bertopic model ",
    description="A simple API that use NLP model to predict topics",
    version="0.1",
)

#load the sentiment model

model = BERTopic.load("models/topicmodel.pkl")

# cleaning the data
def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # remove numbers
    
    # Remove punctuation from text
    text = "".join([c for c in text if c not in punctuation])

        
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
        
    # Return a list of words
    return text

@app.post("/predict-review")
def predict_topics(review: str):
    """
    A simple function that receive a  content and predict the topic of the content.
    :param review:
    :return: prediction, probabilities
    """
    # clean the review
    cleaned_review = text_cleaning(review)
    
    # erform prediction
    similar_topics, similarity = model.find_topics(cleaned_review, top_n=1)
    return similar_topics[0]


@app.get("/get_topics_with_coherence")
def perform_action():
    # Call the model or perform the desired action
    file_ops,result = model.get_topics_with_coherence()  # Replace with the appropriate method or action

    # Return the result as JSON
    return {"result": result}

@app.get("/")
async def root():
    return ("message Hi working:")
if __name__ == "__main__":
    uvicorn.run("main:app")
