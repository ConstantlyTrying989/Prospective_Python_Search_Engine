"""
Test preprocessing pipeline.
"""

import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from loader import DocumentLoader
from preprocessing import TextPreprocessor


def main():
    # Load documents
    loader = DocumentLoader('data/raw_texts')
    documents = loader.load_documents()
    
    print(f"Loaded {len(documents)} documents")
    print()
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(use_stemming=True, remove_stopwords=True)
    
    # Test on first document (just first 500 chars)
    sample_text = documents[0]['content'][:500]
    print("ORIGINAL TEXT (first 500 chars):")
    print(sample_text)
    print("\n" + "="*60 + "\n")
    
    # Preprocess
    tokens = preprocessor.preprocess(sample_text)
    
    print(f"PROCESSED TOKENS ({len(tokens)} tokens):")
    print(tokens[:50])  # First 50 tokens
    print()
    
    # Preprocess all documents
    print("Processing all documents...")
    all_contents = [doc['content'] for doc in documents]
    processed_docs = preprocessor.preprocess_documents(all_contents)
    
    print(f"Total documents processed: {len(processed_docs)}")
    print(f"Average tokens per document: {sum(len(d) for d in processed_docs) / len(processed_docs):.0f}")
    
    # Show stats for each document
    print("\nDocument statistics:")
    for i, (doc, tokens) in enumerate(zip(documents, processed_docs)):
        print(f"  {doc['title']}: {len(tokens)} tokens")


if __name__ == '__main__':
    main()