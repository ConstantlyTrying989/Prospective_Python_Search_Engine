"""
Test script for document loader.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.loader import DocumentLoader


def main():
    # Initialize loader - USE CORRECT PATH (no ../)
    loader = DocumentLoader('data/raw_texts')  # ← Changed from '../data/raw_texts'
    
    # Load documents
    print("Loading documents...\n")
    
    try:
        documents = loader.load_documents()
    except Exception as e:
        print(f"ERROR: {e}")
        print("\nMake sure:")
        print("1. You have .txt files in data/raw_texts folder")
        print("2. You're running from project root")
        return
    
    print(f"\n{'='*60}")
    print(f"Total documents loaded: {loader.get_document_count()}")
    print(f"{'='*60}\n")
    
    # Show document info
    for i, doc in enumerate(documents, 1):
        print(f"{i}. {doc['title']}")
        print(f"   Characters: {len(doc['content']):,}")
        print(f"   Preview: {doc['content'][:100].strip()}...")
        print()


if __name__ == '__main__':
    main()