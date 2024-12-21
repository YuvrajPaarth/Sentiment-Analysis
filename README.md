
# Sentiment Analysis of IMDb Reviews

This project implements a sentiment analysis system using Python to classify IMDb movie reviews as either positive or negative. The model is trained on a dataset of 50,000 IMDb reviews and utilizes word embeddings for natural language processing.

## Features

	•	Data Preprocessing: Text data is tokenized, encoded, and padded for uniformity.
	•	Word Embedding: Words are mapped to dense vectors to capture semantic meaning.
	•	Binary Classification: The model classifies reviews as either positive or negative.
	•	Custom Predictions: Test the model with new reviews for sentiment classification.

## Prerequisites

	•	Python 3.x
	•	TensorFlow
	•	Required Python libraries: pandas, numpy, scikit-learn, and opendatasets.

## Installation

	1.	Clone the Repository

git clone https://github.com/yourusername/sentiment-analysis-imdb.git
cd sentiment-analysis-imdb


	2.	Install the Required Python Packages

pip install tensorflow pandas numpy scikit-learn opendatasets


	3.	Download the Dataset
The project uses the IMDb Dataset of 50K Movie Reviews.
Run the following in Python to download the dataset:

import opendatasets as od
od.download("https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")



## Usage

Preprocessing and Model Training

import opendatasets as od
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from sklearn.model_selection import train_test_split

## Load dataset
file = pd.read_csv('imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')

## Split data
imdb_reviews, test_reviews = train_test_split(file, test_size=0.2, random_state=42)

## Encode labels
def encode_sentiments(sentiment):
    return 1 if sentiment == 'positive' else 0

train_data, train_label = imdb_reviews['review'], imdb_reviews['sentiment'].apply(encode_sentiments)
test_data, test_label = test_reviews['review'], test_reviews['sentiment'].apply(encode_sentiments)

## Tokenize and pad sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data)

train_data = tokenizer.texts_to_sequences(train_data)
test_data = tokenizer.texts_to_sequences(test_data)

vocab_size = len(tokenizer.word_index) + 1
max_length = 500

train_data = pad_sequences(train_data, maxlen=max_length, padding='post')
test_data = pad_sequences(test_data, maxlen=max_length, padding='post')

## Define the model
model = Sequential([
    Embedding(vocab_size, 16),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data, train_label, epochs=30, batch_size=512, validation_data=(test_data, test_label))

Evaluate and Test the Model

## Evaluate the model
loss, accuracy = model.evaluate(test_data, test_label)
print(f"Test Accuracy: {accuracy:.2f}")

## Test with a custom review
user_review = "The movie was fantastic with great acting and direction!"
user_review_seq = tokenizer.texts_to_sequences([user_review])
user_review_padded = pad_sequences(user_review_seq, maxlen=max_length, padding='post')

prediction = model.predict(user_review_padded)
print("Positive sentiment" if prediction > 0.5 else "Negative sentiment")

## Results

	•	Accuracy: Achieved 87.19% accuracy on the test dataset.
	•	Model Performance: The model generalizes well on unseen reviews.

## Dataset

	•	The IMDb Dataset of 50K Movie Reviews is available here.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## Contact

If you have any questions or suggestions, feel free to reach out at yuvraj.works1@gmail.com.
