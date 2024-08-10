import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load and preprocess the dataset."""
    df = pd.read_csv(file_path, encoding='latin-1')
    df = df.rename(columns={'v1': 'label', 'v2': 'message'})
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

def preprocess_data(df):
    """Prepare the data for training and testing."""
    X = df['message']
    y = df['label']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_pipeline():
    """Create a pipeline with TF-IDF Vectorizer and Naive Bayes Classifier."""
    return Pipeline([
        ('tfidf', TfidfVectorizer()),  # Convert text to TF-IDF features
        ('nb', MultinomialNB())        # Apply Naive Bayes classification
    ])

def evaluate_model(y_test, y_pred):
    """Evaluate and print model performance metrics."""
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    # Define file path (update this to your file path)
    file_path = 'spam.csv'
    
    # Load and preprocess data
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Build and train model
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = pipeline.predict(X_test)
    evaluate_model(y_test, y_pred)

if _name_ == "_main_":
    main()