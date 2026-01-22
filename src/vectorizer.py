"""
TF-IDF vectorizer for document search.
"""

import numpy as np
from typing import List, Dict
from collections import Counter
import math


class TFIDFVectorizer:
    """Builds TF-IDF vectors from preprocessed documents."""
    
    def __init__(self):
        """Initialize vectorizer."""
        self.vocabulary: Dict[str, int] = {}  # term -> index mapping
        self.idf_values: np.ndarray = None     # IDF values for each term
        self.num_documents: int = 0
    
    def fit(self, documents: List[List[str]]) -> None:
        """
        Build vocabulary and compute IDF values.
        
        Args:
            documents: List of tokenized documents (list of token lists)
        """
        self.num_documents = len(documents)
        
        # Build vocabulary
        all_terms = set()
        for doc in documents:
            all_terms.update(doc)
        
        # Create term -> index mapping (sorted for consistency)
        self.vocabulary = {term: idx for idx, term in enumerate(sorted(all_terms))}
        
        # Compute document frequency for each term
        doc_freq = Counter()
        for doc in documents:
            unique_terms = set(doc)
            for term in unique_terms:
                doc_freq[term] += 1
        
        # Compute IDF: log(N / df(t))
        vocab_size = len(self.vocabulary)
        self.idf_values = np.zeros(vocab_size)
        
        for term, idx in self.vocabulary.items():
            df = doc_freq[term]
            # Add smoothing to avoid division by zero
            self.idf_values[idx] = math.log((self.num_documents + 1) / (df + 1))
        
        print(f"Vocabulary size: {vocab_size}")
        print(f"Documents: {self.num_documents}")
    
    def transform(self, document: List[str]) -> np.ndarray:
        """
        Convert single document to TF-IDF vector.
        
        Args:
            document: Tokenized document (list of tokens)
            
        Returns:
            TF-IDF vector as numpy array
        """
        # Initialize zero vector
        vector = np.zeros(len(self.vocabulary))
        
        # Count term frequencies
        term_freq = Counter(document)
        doc_length = len(document)
        
        if doc_length == 0:
            return vector
        
        # Compute TF-IDF for each term in document
        for term, count in term_freq.items():
            if term in self.vocabulary:
                idx = self.vocabulary[term]
                tf = count / doc_length  # Normalized term frequency
                vector[idx] = tf * self.idf_values[idx]
        
        return vector
    
    def fit_transform(self, documents: List[List[str]]) -> np.ndarray:
        """
        Fit vocabulary and transform documents in one step.
        
        Args:
            documents: List of tokenized documents
            
        Returns:
            Document-term matrix (num_docs x vocab_size)
        """
        self.fit(documents)
        
        # Transform all documents
        matrix = np.zeros((self.num_documents, len(self.vocabulary)))
        for i, doc in enumerate(documents):
            matrix[i] = self.transform(doc)
        
        return matrix