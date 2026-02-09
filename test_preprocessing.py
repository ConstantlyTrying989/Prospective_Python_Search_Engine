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
    print("="*70)
    print("TEXT PREPROCESSING TEST")
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
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(use_stemming=True, remove_stopwords=True)
    
    # Test on sample text
    sample_text = documents[0]['content'][:500]
    print("="*70)
    print("ORIGINAL TEXT (first 500 chars):")
    print("="*70)
    print(sample_text)
    print("\n" + "="*70 + "\n")
    
    tokens = preprocessor.preprocess(sample_text)
    print(f"PROCESSED TOKENS ({len(tokens)} tokens):")
    print("="*70)
    print(tokens[:50])  # Show first 50 tokens
    print()
    
    # Process all documents
    print("="*70)
    print("PROCESSING ALL DOCUMENTS...")
    print("="*70)
    all_contents = [doc['content'] for doc in documents]
    processed_docs = preprocessor.preprocess_documents(all_contents)
    
    avg_tokens = sum(len(d) for d in processed_docs) / len(processed_docs)
    print(f"\n✅ Total documents processed: {len(processed_docs)}")
    print(f"✅ Average tokens per document: {avg_tokens:,.0f}")
    
    print("\n" + "="*70)
    print("DOCUMENT STATISTICS:")
    print("="*70)
    for doc, tokens in zip(documents, processed_docs):
        print(f"  • {doc['title']}: {len(tokens):,} tokens")
    
    print("\n" + "="*70)
    print("✅ Test completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()