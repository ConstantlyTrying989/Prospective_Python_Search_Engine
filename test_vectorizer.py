"""
Test TF-IDF vectorizer.
"""

import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import numpy as np
from src.loader import DocumentLoader
from src.preprocessing import TextPreprocessor
from src.vectorizer import TFIDFVectorizer


def main():
    loader = DocumentLoader('data/raw_texts')  # ← Relative to project root
    
    try:
        documents = loader.load_documents()
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    print(f"\nLoaded {len(documents)} documents\n")
    
    preprocessor = TextPreprocessor(use_stemming=True, remove_stopwords=True)
    all_contents = [doc['content'] for doc in documents]
    processed_docs = preprocessor.preprocess_documents(all_contents)
    
    print("Building TF-IDF vectors...")
    vectorizer = TFIDFVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_docs)
    
    print(f"\nTF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"(documents x vocabulary size)\n")
    
    doc_idx = 0
    doc_vector = tfidf_matrix[doc_idx]
    nonzero_indices = np.where(doc_vector > 0)[0]
    top_indices = nonzero_indices[np.argsort(-doc_vector[nonzero_indices])][:10]
    
    print(f"Top 10 terms in '{documents[doc_idx]['title']}':")
    reverse_vocab = {idx: term for term, idx in vectorizer.vocabulary.items()}
    for idx in top_indices:
        term = reverse_vocab[idx]
        score = doc_vector[idx]
        print(f"  {term}: {score:.4f}")


if __name__ == '__main__':
    main()