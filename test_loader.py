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
    print("="*70)
    print("DOCUMENT LOADER TEST")
    print("="*70)
    print()
    
    # Initialize loader with relative path (loader.py will handle converting to absolute)
    loader = DocumentLoader('data/raw_texts')
    
    # Load documents
    print("Loading documents...\n")
    
    try:
        documents = loader.load_documents()
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have .txt files in data/raw_texts folder")
        print("2. Check that the folder structure is:")
        print("   Prospective_Python_Search_Engine/")
        print("   ├── data/")
        print("   │   └── raw_texts/")
        print("   │       └── (your .txt files)")
        print("   └── test_loader.py")
        return
    
    print(f"\n{'='*70}")
    print(f"✅ Total documents loaded: {loader.get_document_count()}")
    print(f"{'='*70}\n")
    
    # Show document info
    for i, doc in enumerate(documents, 1):
        print(f"{i}. {doc['title']}")
        print(f"   Characters: {len(doc['content']):,}")
        print(f"   Preview: {doc['content'][:100].strip()}...")
        print()
    
    print("="*70)
    print("✅ Test completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()