"""
Interactive search engine demo.
"""

import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.loader import DocumentLoader
from src.search import SearchEngine


def main():
    print("="*70)
    print("DOCUMENT SEARCH ENGINE - Interactive Demo")
    print("="*70)
    print()
    
    # Load documents
    print("Loading documents...")
    loader = DocumentLoader('data/raw_texts')
    documents = loader.load_documents()
    print()
    
    # Build search index
    search_engine = SearchEngine()
    search_engine.index_documents(documents)
    
    # Interactive search loop
    print("="*70)
    print("Enter your search queries (or 'quit' to exit)")
    print("="*70)
    print()
    
    while True:
        # Get query from user
        query = input("Search query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nExiting search engine. Goodbye!")
            break
        
        if not query:
            print("Please enter a query.\n")
            continue
        
        print()
        
        # Perform search
        results = search_engine.search(query, top_k=5)
        search_engine.print_results(query, results)
        
        print()


if __name__ == '__main__':
    main()