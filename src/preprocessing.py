"""
Text preprocessing module for search engine.
Handles cleaning, tokenization, and stopword removal.
"""

import re
import string
from typing import List
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download required NLTK data (run once)
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TextPreprocessor:
    """Preprocesses text documents for search indexing."""
    
    def __init__(self, use_stemming: bool = True, remove_stopwords: bool = True):
        """
        Initialize preprocessor.
        
        Args:
            use_stemming: Apply Porter stemming to tokens
            remove_stopwords: Remove English stopwords
        """
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.stemmer = PorterStemmer() if use_stemming else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned lowercase text
        """
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize and filter text.
        
        Args:
            text: Cleaned text string
            
        Returns:
            List of processed tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove punctuation and short tokens
        tokens = [
            token for token in tokens 
            if token not in string.punctuation and len(token) > 2
        ]
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming
        if self.use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        return tokens
    
    def preprocess(self, text: str) -> List[str]:
        """
        Full preprocessing pipeline.
        
        Args:
            text: Raw text string
            
        Returns:
            List of processed tokens
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        return tokens
    
    def preprocess_documents(self, documents: List[str]) -> List[List[str]]:
        """
        Preprocess multiple documents.
        
        Args:
            documents: List of raw text strings
            
        Returns:
            List of token lists
        """
        return [self.preprocess(doc) for doc in documents]