"""
Test preprocessing pipeline.
"""

import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.loader import DocumentLoader
from src.preprocessing import TextPreprocessor


def main():
    loader = DocumentLoader('data/raw_texts')  # ← Relative to project root
    
    try:
        documents = loader.load_documents()
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    print(f"\nLoaded {len(documents)} documents\n")
    
    preprocessor = TextPreprocessor(use_stemming=True, remove_stopwords=True)
    
    sample_text = documents[0]['content'][:500]
    print("ORIGINAL TEXT (first 500 chars):")
    print(sample_text)
    print("\n" + "="*60 + "\n")
    
    tokens = preprocessor.preprocess(sample_text)
    print(f"PROCESSED TOKENS ({len(tokens)} tokens):")
    print(tokens[:50])
    print()
    
    all_contents = [doc['content'] for doc in documents]
    processed_docs = preprocessor.preprocess_documents(all_contents)
    
    print(f"Total documents processed: {len(processed_docs)}")
    print(f"Average tokens per document: {sum(len(d) for d in processed_docs) / len(processed_docs):.0f}")
    
    print("\nDocument statistics:")
    for doc, tokens in zip(documents, processed_docs):
        print(f"  {doc['title']}: {len(tokens)} tokens")


if __name__ == '__main__':
    main()