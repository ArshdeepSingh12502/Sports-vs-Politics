import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, classification_report, confusion_matrix)
import re
import json
import pickle
from typing import Dict, Tuple, Any
import time

class TextClassifier:
      
    def __init__(self):
        self.vectorizers = {}
        self.models = {}
        self.results = {}
        
    def preprocess_text(self, text: str) -> str:
        # Convert to lowercase , remove special characters and whitespaces
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    def extract_features_ngrams(self, X_train, X_test, ngram_range=(1, 2), max_features=1000):   #using ngram for feature extraction
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            preprocessor=self.preprocess_text,
            stop_words='english'
        )
        
        X_train_ngrams = vectorizer.fit_transform(X_train)
        X_test_ngrams = vectorizer.transform(X_test)
        
        return X_train_ngrams, X_test_ngrams, vectorizer
    
    def train_naive_bayes(self, X_train, y_train, X_test, y_test, feature_name: str):
        print(f"\nTraining Naive Bayes with {feature_name}\n")
        
        start_time = time.time()
        
        # Train model
        nb_model = MultinomialNB()
        nb_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = nb_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='sports')
        recall = recall_score(y_test, y_pred, pos_label='sports')
        f1 = f1_score(y_test, y_pred, pos_label='sports')
        
        training_time = time.time() - start_time
        
        # Store results
        result_key = f"NaiveBayes_{feature_name}"
        self.results[result_key] = {
            'accuracy': accuracy,
            'training_time': training_time,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        self.models[result_key] = nb_model
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Training Time: {training_time:.4f}s")
        
        return nb_model, self.results[result_key]
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test, feature_name: str):
        print(f"\nTraining Logistic Regression with {feature_name}\n")
        
        start_time = time.time()
        
        # Train model
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = lr_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='sports')
        recall = recall_score(y_test, y_pred, pos_label='sports')
        f1 = f1_score(y_test, y_pred, pos_label='sports')
        
        training_time = time.time() - start_time
        
        # Store results
        result_key = f"LogisticRegression_{feature_name}"
        self.results[result_key] = {
            'accuracy': accuracy,
            'training_time': training_time,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        self.models[result_key] = lr_model
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Training Time: {training_time:.4f}s")
        
        return lr_model, self.results[result_key]
    
    def train_svm(self, X_train, y_train, X_test, y_test, feature_name: str):
        print(f"\nTraining SVM with {feature_name}\n")
        
        start_time = time.time()
        
        # Train model
        svm_model = SVC(kernel='linear', random_state=42)
        svm_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = svm_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='sports')
        recall = recall_score(y_test, y_pred, pos_label='sports')
        f1 = f1_score(y_test, y_pred, pos_label='sports')
        
        training_time = time.time() - start_time
        
        # Store results
        result_key = f"SVM_{feature_name}"
        self.results[result_key] = {
            'accuracy': accuracy,
            'training_time': training_time,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        self.models[result_key] = svm_model
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Training Time: {training_time:.4f}s")
        
        return svm_model, self.results[result_key]
    
    def save_results(self, filepath: str):
        with open(filepath, 'w') as f:
            # Convert results to JSON format
            results_to_save = {}
            for key, value in self.results.items():
                results_to_save[key] = {
                    k: v for k, v in value.items() 
                    if k != 'classification_report'  # Skip non-JSON
                }
                results_to_save[key]['classification_report'] = str(value['classification_report'])
            
            json.dump(results_to_save, f, indent=2)
        
        print(f"\nResults saved to {filepath}")

    def save_models(self, directory: str):
        """Save trained models to disk"""
        for name, model in self.models.items():
            filepath = f"{directory}/{name}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        
        print(f"Models saved to {directory}/")

def main():    
    
    train_df = pd.read_csv('data/train.csv')    #loading data set
    test_df = pd.read_csv('data/test.csv')
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # preprocessing data
    X_train = train_df['text'].values
    y_train = train_df['label'].values
    X_test = test_df['text'].values
    y_test = test_df['label'].values
    
    # Initialize classifier
    classifier = TextClassifier()
            
    print("Extracting N-gram features...")
    X_train_ngrams, X_test_ngrams, ngram_vec = classifier.extract_features_ngrams(X_train, X_test)
    classifier.vectorizers['Ngrams'] = ngram_vec
    
    print(f"N-grams features shape: {X_train_ngrams.shape}")
    
    # Applying Naive Bayes,Logistic Regression and SVM
    classifier.train_naive_bayes(X_train_ngrams, y_train, X_test_ngrams, y_test, "Ngrams")
    classifier.train_logistic_regression(X_train_ngrams, y_train, X_test_ngrams, y_test, "Ngrams")
    classifier.train_svm(X_train_ngrams, y_train, X_test_ngrams, y_test, "Ngrams")

    classifier.save_results('results/classification_results.json')
    
    # Find best model
    best_model = max(classifier.results.items(), key=lambda x: x[1]['accuracy'])
    print(f"BEST PERFORMING MODEL: {best_model[0]}")

if __name__ == "__main__":
    main()
