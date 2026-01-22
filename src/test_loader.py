"""
Test script for document loader.
"""

from loader import DocumentLoader


def main():
    # Initialize loader - use path relative to src folder
    loader = DocumentLoader('../data/raw_texts')  # Added '../'
    
    # Load documents
    print("Loading documents...\n")
    documents = loader.load_documents()
    
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