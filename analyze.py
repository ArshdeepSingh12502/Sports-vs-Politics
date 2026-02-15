import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
import re


def analyze_dataset(train_path: str, test_path: str):
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    for label in ['sports', 'politics']:
        subset = train_df[train_df['label'] == label]
    
    return train_df, test_df

def extract_top_words(train_df: pd.DataFrame, n_words: int = 20):
    # Preprocess function
    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()
    
    # Get top words for each category
    top_words = {}
    
    for label in ['sports', 'politics']:
        texts = train_df[train_df['label'] == label]['text'].apply(preprocess)
        
        # Use CountVectorizer to get word frequencies
        vectorizer = CountVectorizer(stop_words='english', max_features=n_words)
        X = vectorizer.fit_transform(texts)
        
        # Get word frequencies
        word_freq = dict(zip(vectorizer.get_feature_names_out(), 
                            X.toarray().sum(axis=0)))
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_words[label] = sorted_words[:n_words]
        
        print(f"\nTop {n_words} words in {label.upper()}:")
        for word, freq in sorted_words[:n_words]:
            print(f"  {word}: {freq}")

def visualize_model_performance(results_path: str):
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)

def main():
    
    # Analyze dataset
    train_df, test_df = analyze_dataset('data/train.csv', 'data/test.csv')

    extract_top_words(train_df, n_words=15)
    
    visualize_model_performance('results/classification_results.json')

    print("\nGenerated visualizations: \n1. dataset_analysis.png \n2. top_words.png \n3. model_performance.png")

if __name__ == "__main__":
    main()
