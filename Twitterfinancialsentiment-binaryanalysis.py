# -*- coding: utf-8 -*-
"""
Frankfurt School of Finance & Management, Subject to Copyright
Created on Mon Feb 17 12:57:57 2025
Author: Abdul Malik ()
Description: Twitter Financial Sentiment Analysis
This code is submitted as part of DAMLF course for Master of Finance program at FSFM. the script performs binary classification on financial tweets 

Assumptions and Notes:
    1. Update the filepath before executing
    2. Dataset has been cleaned (converted to lowercase and removing special characters) before stemming
    3. Max Iter has been set to 5000.
    4. Random Forest used as algorith of choice foe step 13
    5. Where instruction specified stronger regularisation C = 0.5 has been used. C = 1 being default and less than 1 being stronger.
"""

#=========== IMPORT LIBRARIES =================
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import re

##############################################################################
print("Step 1: IMPORT DATASET")

# Load the Twitter Financial News dataset
file_path = r"C:\Users\abdul\Desktop\TwitterFinancialSentiment.csv"
df = pd.read_csv(file_path)
print("Dataset loaded:", df.shape)

##############################################################################
print("\nStep 2: DISPLAY UNIQUE LABELS")

# Labels: 0 (Bearish), 1 (Bullish), 2 (Neutral)
print("Unique labels in dataset:", df['label'].unique())

##############################################################################
print("\nStep 3: DROP NEUTRAL SENTIMENT (LABEL = 2)")

# Drop rows where the label is equal to 2 to create binary classification
df = df.drop(df[df["label"] == 2].index)
print("Remaining labels in dataset:", df['label'].unique())
print(df.shape)


##############################################################################
print("\nStep 4: APPLY STEMMING")

# Initialize Porter Stemmer for word stemming
stemmer = PorterStemmer()

def clean_text(text):
    """
    Preprocess text by converting to lowercase and removing special characters
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    return text

# Apply text cleaning and stemming to the 'text' column
df['cleaned_text'] = df['text'].apply(clean_text)
df['stemmed_text'] = df['cleaned_text'].apply(
    lambda x: ' '.join([stemmer.stem(word) for word in x.split()])
)
print("Stemming applied.")
print("Raw Text:\n", df['text'].head())
print("Stemmed Text:\n", df['stemmed_text'].head())

##############################################################################
print("\nStep 5: CREATE MATRIX X")

# Create feature matrix X containing stemmed text
X = df['stemmed_text']
print("Matrix X created with stemmed text")

##############################################################################
print("\nStep 6: CREATE VECTOR Y")

# Create target vector y containing labels
y = df["label"]
print("Vector y created with labels")

##############################################################################
print("\nStep 7: SPLIT DATA")

# Split data into train (75%) and test (25%) sets using StratifiedShuffleSplit
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=37)
for train_index, test_index in splitter.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
print("Split Successful.")

##############################################################################
print("\nStep 8: TF-IDF VECTORIZATION")

# Custom stop words list as specified in the instructions
mystoplist = ["the", "a", "in", "to", "of", "from", "and", "profit", "for", "it", "year"]

# Create TF-IDF vectorizer with unigrams and bigrams, using custom stop words
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=mystoplist)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("TF-IDF Matrix Shape (Train):", X_train_tfidf.shape)
print("TF-IDF Matrix Shape (Test):", X_test_tfidf.shape)

# Performance evaluation function
def print_performance(model_name, model, X_train, X_test):
    """
    Print train and test accuracy scores for a model
    
    """
    print(f"{model_name}:")
    print(f"  Train accuracy: {accuracy_score(y_train, model.predict(X_train)):.4f}")
    print(f"  Test accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}\n")

##############################################################################
print("\nStep 9: LOGISTIC REGRESSION")

# Logistic Regression with elastic net and stronger regularization
logreg = LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    l1_ratio=0.5,
    C=0.5,  # Stronger regularization (default C=1)
    max_iter=5000)
logreg.fit(X_train_tfidf, y_train)
print_performance("Logistic Regression", logreg, X_train_tfidf, X_test_tfidf)

##############################################################################
print("\nStep 10: NAIVE BAYES")

# Multinomial Naive Bayes with default settings
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
print_performance("Naive Bayes", nb, X_train_tfidf, X_test_tfidf)

##############################################################################
print("\nStep 11: SVC")

# Support Vector Classification with stronger regularization
svc = SVC(kernel='linear', C=0.5, max_iter=5000)
svc.fit(X_train_tfidf, y_train)
print_performance("SVC", svc, X_train_tfidf, X_test_tfidf)

##############################################################################
print("\nStep 12: NEURAL NETWORK")

# Neural Network with specified parameters
mlp = MLPClassifier(
    hidden_layer_sizes=(10, 10),  # 2 hidden layers with 10 units each
    activation='relu',
    alpha=0.01,  # regularization parameter
    max_iter=5000
)
mlp.fit(X_train_tfidf, y_train)
print_performance("Neural Network", mlp, X_train_tfidf, X_test_tfidf)

##############################################################################
print("\nStep 13: GRID SEARCH")

# Define parameter grid for Random Forest tuning
param_grid = {
    'n_estimators': [50, 100, 200, 300],  # Number of trees
    'max_depth': [None, 5, 10, 15, 20, 25]  # Maximum depth of trees
}

# Initialize Random Forest model
rf = RandomForestClassifier()

# Setup 5-fold cross-validation
cv = StratifiedKFold(n_splits=5)

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(rf, param_grid, cv=cv, n_jobs=-1, scoring='accuracy', return_train_score=True)
grid_search.fit(X_train_tfidf, y_train)

# Save and display results
allresults = pd.DataFrame(grid_search.cv_results_)
print("Hyperparameter Tuning for Random Forest")
print("Best Parameters:", grid_search.best_params_)
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# ========== EVALUATE RANDOM FOREST MODEL PERFORMANCE ==========

# Fit the best estimator from grid search on the training data
best_rf_model = grid_search.best_estimator_
best_rf_model.fit(X_train_tfidf, y_train)

# Calculate and print performance on the training set
train_accuracy = accuracy_score(y_train, best_rf_model.predict(X_train_tfidf))
print("\nRandom Forest Model Performance:")
print(f"  Train accuracy: {train_accuracy:.4f}")

# Calculate and print performance on the test set
rf_test_acc = accuracy_score(y_test, best_rf_model.predict(X_test_tfidf))
print(f"  Test accuracy: {rf_test_acc:.4f}")

# Compare with Naive Bayes performance
nb_test_acc = accuracy_score(y_test, nb.predict(X_test_tfidf))

print("\nComparison with Task 10 (Naive Bayes):")
print(f"- Naive Bayes Test Accuracy: {nb_test_acc:.4f}")
print(f"- Random Forest Test Accuracy: {rf_test_acc:.4f}")
print(f"Accuracy Improvement: {(rf_test_acc - nb_test_acc):.4f}")


