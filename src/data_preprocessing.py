# src/data_preprocessing.py

import pandas as pd
import re

def load_data(path: str) -> pd.DataFrame:
    
    df = pd.read_csv(path, parse_dates=['Date received', 'Date sent to company'])
    return df

def filter_products(df: pd.DataFrame, products: list) -> pd.DataFrame:
    
    filtered = df[df['Product'].isin(products)].copy()
    filtered = filtered[filtered['Consumer complaint narrative'].notna()]
    return filtered

def clean_text(text: str) -> str:
    
    if not isinstance(text, str):
        return ""
    text = text.lower()  # lowercase
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # normalize spaces
    
    # Remove common boilerplate phrases
    boilerplate_phrases = [
        "i am writing to file a complaint", 
        "i am writing to complain",
        "please investigate this matter",
        "i want to file a complaint",
        "thank you for your time"
    ]
    for phrase in boilerplate_phrases:
        text = text.replace(phrase, "")
    
    return text

def add_narrative_stats(df: pd.DataFrame) -> pd.DataFrame:
    
    df['narrative_char_count'] = df['Consumer complaint narrative'].apply(lambda x: len(str(x)))
    df['narrative_word_count'] = df['Consumer complaint narrative'].apply(lambda x: len(str(x).split()))
    return df

def save_data(df: pd.DataFrame, path: str):
    
    df.to_csv(path, index=False)
