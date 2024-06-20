# sentiment_analysis.py

import nltk
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random

# Download necessary NLTK data
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')

# Load the movie reviews dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the documents
random.shuffle(documents)

# Define a function to preprocess the documents
def preprocess_words(words):
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in words if word.isalpha() and word not in stop_words]
    return words

# Preprocess the documents
documents = [(preprocess_words(doc), category) for doc, category in documents]

# Convert the documents to strings
document_strings = [" ".join(doc) for doc, category in documents]

# Create a CountVectorizer
vectorizer = CountVectorizer(max_features=2000)

# Fit and transform the data
X = vectorizer.fit_transform(document_strings).toarray()

# Get the labels
y = [category for doc, category in documents]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='pos')
recall = recall_score(y_test, y_pred, pos_label='pos')
f1 = f1_score(y_test, y_pred, pos_label='pos')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Define a function to predict sentiment
def predict_sentiment(text):
    text = preprocess_words(word_tokenize(text))
    text_string = " ".join(text)
    text_features = vectorizer.transform([text_string]).toarray()
    prediction = classifier.predict(text_features)
    return prediction[0]

# Ask the user for a movie review and predict its sentiment
user_review = input("Enter a movie review: ")
print("The sentiment of your review is:", predict_sentiment(user_review))
