"""
Test TF-IDF vectorizer.
"""

import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import numpy as np
from loader import DocumentLoader
from preprocessing import TextPreprocessor
from vectorizer import TFIDFVectorizer


def main():
    # Load and preprocess
    loader = DocumentLoader('data/raw_texts')
    documents = loader.load_documents()
    
    preprocessor = TextPreprocessor(use_stemming=True, remove_stopwords=True)
    all_contents = [doc['content'] for doc in documents]
    processed_docs = preprocessor.preprocess_documents(all_contents)
    
    print(f"Loaded {len(documents)} documents\n")
    
    # Fit TF-IDF
    print("Building TF-IDF vectors...")
    vectorizer = TFIDFVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_docs)
    
    print(f"\nTF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"(documents x vocabulary size)\n")
    
    # Show top terms by TF-IDF for first document
    doc_idx = 0
    doc_vector = tfidf_matrix[doc_idx]
    
    # Get non-zero indices
    nonzero_indices = np.where(doc_vector > 0)[0]
    
    # Sort by TF-IDF score
    top_indices = nonzero_indices[np.argsort(-doc_vector[nonzero_indices])][:10]
    
    print(f"Top 10 terms in '{documents[doc_idx]['title']}':")
    reverse_vocab = {idx: term for term, idx in vectorizer.vocabulary.items()}
    for idx in top_indices:
        term = reverse_vocab[idx]
        score = doc_vector[idx]
        print(f"  {term}: {score:.4f}")


if __name__ == '__main__':
    main()