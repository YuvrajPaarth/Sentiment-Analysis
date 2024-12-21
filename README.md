Sentiment Analysis of IMDb Movie Reviews

This project uses a neural network to perform sentiment analysis on IMDb movie reviews. The dataset was obtained from Kaggle and preprocessed to prepare it for a deep learning model built using TensorFlow/Keras.

Dataset

Sources

	1.	IMDb Dataset of 50k Movie Reviews: Contains movie reviews and their associated sentiments (positive or negative).
	2.	Word Index Dataset: Maps words to unique indexes for text encoding.

Structure

	•	Columns in the IMDb dataset:
	•	review: Text of the movie review.
	•	sentiment: Sentiment (positive or negative).
	•	Columns in the Word Index dataset:
	•	Words: Unique words in the dataset.
	•	Indexes: Corresponding unique indexes for each word.

Project Workflow

1. Environment Setup

Install the necessary libraries:

pip install opendatasets
pip install tensorflow

2. Dataset Download

Using opendatasets, the IMDb reviews and word index datasets were downloaded from Kaggle.

import opendatasets as od

od.download("https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
od.download("https://www.kaggle.com/datasets/virenmohanlal/word-indexes")

3. Data Preprocessing

	•	Split Dataset:
	•	The IMDb dataset was split into training (80%) and testing (20%) sets using train_test_split.
	•	Word Indexing:
	•	Mapped each word in the review to its corresponding index using the Word Index dataset.
	•	Sentiment Encoding:
	•	Converted sentiments into binary labels (positive: 1, negative: 0).
	•	Padding:
	•	Padded reviews to ensure equal length of 500 words for input consistency.

train_data=keras.preprocessing.sequence.pad_sequences(train_data, value=padding_value, padding='post', maxlen=500)
test_data=keras.preprocessing.sequence.pad_sequences(test_data, value=padding_value, padding='post', maxlen=500)

4. Model Building

	•	Embedding Layer:
Converts word indexes into dense vectors of fixed size (16).
	•	Global Average Pooling:
Reduces the dimensionality of the embedding vectors.
	•	Dense Layers:
	•	16 units with ReLU activation.
	•	1 unit with Sigmoid activation for binary classification.

model = keras.Sequential([
    keras.layers.Embedding(max_word_index + 1, 16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

	•	Compilation:
	•	Optimizer: Adam
	•	Loss: Binary Crossentropy
	•	Metrics: Accuracy

5. Model Training

The model was trained on the training data for 30 epochs with a batch size of 512. Validation was performed on the test set.

history = model.fit(train_data, train_label, epochs=30, batch_size=512, validation_data=(test_data, test_label))

6. Evaluation

The trained model achieved an accuracy of 87.19% on the test dataset.

loss, accuracy = model.evaluate(test_data, test_label)
print(f"Accuracy: {accuracy:.2f}, Loss: {loss:.2f}")

7. Prediction

The model can predict the sentiment of a randomly chosen review from the test set.

user_review = test_data[index]
user_review = np.array([user_review])
if (model.predict(user_review) > 0.5).astype("int32"):
    print("positive sentiment")
else:
    print("negative sentiment")

Results

The model successfully predicts the sentiment of movie reviews with high accuracy. Predictions on unseen reviews demonstrate the model’s capability for generalization.

Files in the Repository

	1.	IMDB Dataset.csv - Original IMDb dataset.
	2.	word_indexes.csv - Word-to-index mapping dataset.
	3.	README.md - Documentation file (this file).

How to Run the Project

	1.	Clone or download the project.
	2.	Run the script in Google Colab.
	3.	Ensure Kaggle API keys are configured in Colab for dataset download.
	4.	Execute each cell sequentially to train, evaluate, and test the model.

Technologies Used

	•	Python: Programming language.
	•	TensorFlow/Keras: Deep learning framework.
	•	Pandas: Data manipulation.
	•	NumPy: Numerical computations.
	•	scikit-learn: Train/test splitting.

