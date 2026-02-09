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
    print("="*70)
    print("TF-IDF VECTORIZER TEST")
    print("="*70)
    print()
    
    # Load documents
    loader = DocumentLoader('data/raw_texts')
    
    try:
        documents = loader.load_documents()
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print("\nMake sure you have .txt files in data/raw_texts folder")
        return
    
    print(f"\n✅ Loaded {len(documents)} documents\n")
    
    # Preprocess documents
    print("="*70)
    print("PREPROCESSING DOCUMENTS...")
    print("="*70)
    preprocessor = TextPreprocessor(use_stemming=True, remove_stopwords=True)
    all_contents = [doc['content'] for doc in documents]
    processed_docs = preprocessor.preprocess_documents(all_contents)
    print("✅ Preprocessing complete\n")
    
    # Build TF-IDF vectors
    print("="*70)
    print("BUILDING TF-IDF VECTORS...")
    print("="*70)
    vectorizer = TFIDFVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_docs)
    
    print(f"\n✅ TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"   (documents × vocabulary size)\n")
    
    # Analyze first document
    print("="*70)
    print(f"TOP TERMS IN '{documents[0]['title']}':")
    print("="*70)
    
    doc_idx = 0
    doc_vector = tfidf_matrix[doc_idx]
    
    # Get non-zero indices and sort by TF-IDF score
    nonzero_indices = np.where(doc_vector > 0)[0]
    top_indices = nonzero_indices[np.argsort(-doc_vector[nonzero_indices])][:15]
    
    # Create reverse vocabulary mapping
    reverse_vocab = {idx: term for term, idx in vectorizer.vocabulary.items()}
    
    print()
    for rank, idx in enumerate(top_indices, 1):
        term = reverse_vocab[idx]
        score = doc_vector[idx]
        print(f"  {rank:2d}. {term:20s} → {score:.4f}")
    
    # Show vocabulary statistics
    print("\n" + "="*70)
    print("VOCABULARY STATISTICS:")
    print("="*70)
    print(f"  • Total unique terms: {len(vectorizer.vocabulary):,}")
    print(f"  • Documents: {vectorizer.num_documents}")
    print(f"  • Average non-zero terms per document: {np.count_nonzero(tfidf_matrix) / len(documents):,.0f}")
    
    print("\n" + "="*70)
    print("✅ Test completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()