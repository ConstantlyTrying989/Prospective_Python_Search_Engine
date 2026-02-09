"""
Interactive search engine demo.
"""

import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from loader import DocumentLoader
from search import SearchEngine


def main():
    print()
    print("="*70)
    print(" "*15 + "DOCUMENT SEARCH ENGINE")
    print(" "*20 + "Interactive Demo")
    print("="*70)
    print()
    
    # Load documents
    print("📚 Loading documents...")
    try:
        loader = DocumentLoader('data/raw_texts')
        documents = loader.load_documents()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nMake sure you have .txt files in data/raw_texts folder")
        return
    
    print()
    
    # Build search index
    search_engine = SearchEngine()
    search_engine.index_documents(documents)
    
    # Show available documents
    print("="*70)
    print("AVAILABLE DOCUMENTS:")
    print("="*70)
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc['title']}")
    print()
    
    # Interactive search loop
    print("="*70)
    print("SEARCH INTERFACE")
    print("="*70)
    print("Enter your search queries (or 'quit'/'exit'/'q' to exit)")
    print("Try queries like:")
    print("  • 'detective mystery crime'")
    print("  • 'ocean whale adventure'")
    print("  • 'love romance marriage'")
    print("="*70)
    print()
    
    while True:
        # Get query from user
        query = input("🔍 Search query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Exiting search engine. Goodbye!")
            break
        
        if not query:
            print("⚠️  Please enter a query.\n")
            continue
        
        print()
        
        # Perform search
        results = search_engine.search(query, top_k=5)
        
        if not results:
            print("❌ No results found.\n")
            continue
        
        # Print results
        print(f"📊 Found {len(results)} relevant documents for '{query}'")
        print("="*70)
        print()
        
        for result in results:
            print(f"Rank {result['rank']}: {result['title']}")
            print(f"Score: {result['score']:.4f}")
            print(f"Preview: {result['preview']}")
            print("-"*70)
            print()


if __name__ == '__main__':
    main()