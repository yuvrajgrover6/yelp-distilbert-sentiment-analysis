"""
Preprocessing utilities for Yelp sentiment analysis.
This file contains all the data preprocessing functions that can be imported
and used in training and inference notebooks.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple


def load_yelp_data(file_path: str) -> pd.DataFrame:
    """
    Load Yelp dataset from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
    """
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded with shape: {df.shape}")
    return df


def clean_yelp_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Yelp dataset by removing missing values and filtering labels.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    print("Starting data cleaning...")
    
    # Drop rows with missing cleaned_text or sentiment label
    df_clean = df.dropna(subset=["cleaned_text", "star_sentiment"])
    print(f"Removed {len(df) - len(df_clean)} rows with missing values")
    
    # Keep only reviews labeled as positive, negative, or neutral
    df_clean = df_clean[df_clean["star_sentiment"].isin(["positive", "negative", "neutral"])]
    
    # Reset index
    df_clean.reset_index(drop=True, inplace=True)
    
    print(f"Final dataset shape: {df_clean.shape}")
    return df_clean


def encode_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Encode sentiment labels to numerical values.
    
    Args:
        df: DataFrame with sentiment labels
        
    Returns:
        Tuple of (DataFrame with encoded labels, LabelEncoder instance)
    """
    print("Encoding sentiment labels...")
    
    label_encoder = LabelEncoder()
    df_encoded = df.copy()
    df_encoded["label"] = label_encoder.fit_transform(df["star_sentiment"])
    
    # Create label mapping
    label_map = dict(zip(
        label_encoder.classes_, 
        label_encoder.transform(label_encoder.classes_)
    ))
    
    print("Label Encoding Mapping:", label_map)
    return df_encoded, label_encoder


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: DataFrame with text and labels
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        Tuple of (train_texts, val_texts, test_texts, train_labels, val_labels, test_labels)
    """
    print("Splitting data into train/validation/test sets...")
    
    # Split into train (80%) and temp (20%)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df["cleaned_text"], df["label"], 
        test_size=test_size, stratify=df["label"], random_state=random_state
    )
    
    # Split temp into val (10%) and test (10%)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, 
        test_size=0.5, stratify=temp_labels, random_state=random_state
    )
    
    print(f"Train size: {len(train_texts)} ({len(train_texts)/len(df)*100:.1f}%)")
    print(f"Validation size: {len(val_texts)} ({len(val_texts)/len(df)*100:.1f}%)")
    print(f"Test size: {len(test_texts)} ({len(test_texts)/len(df)*100:.1f}%)")
    
    return train_texts, val_texts, test_texts, train_labels.values, val_labels.values, test_labels.values


def visualize_label_distribution(df: pd.DataFrame, title: str = "Label Distribution"):
    """
    Visualize the distribution of sentiment labels.
    
    Args:
        df: DataFrame with sentiment labels
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="star_sentiment", order=["positive", "neutral", "negative"])
    plt.title(title)
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.show()
    
    print("\nLabel Distribution:")
    print(df["star_sentiment"].value_counts())
    print("\nPercentage Distribution:")
    print(df["star_sentiment"].value_counts(normalize=True) * 100)


def preprocess_yelp_data(file_path: str) -> Tuple:
    """
    Complete preprocessing pipeline for Yelp data.
    
    Args:
        file_path: Path to the Yelp dataset CSV
        
    Returns:
        Tuple of processed data splits and label encoder
    """
    # Load data
    df = load_yelp_data(file_path)
    
    # Clean data
    df_clean = clean_yelp_data(df)
    
    # Encode labels
    df_encoded, label_encoder = encode_labels(df_clean)
    
    # Split data
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = split_data(df_encoded)
    
    # Visualize distribution
    visualize_label_distribution(df_clean)
    
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, label_encoder


if __name__ == "__main__":
    # Example usage
    print("Yelp Data Preprocessing Utilities")
    print("Import this module to use preprocessing functions in your notebooks")
