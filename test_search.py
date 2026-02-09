"""
Test search engine functionality.
"""

import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import numpy as np
from src.loader import DocumentLoader
from src.search import SearchEngine


def test_cosine_similarity():
    """Test cosine similarity calculation."""
    print("Testing cosine similarity...")
    
    engine = SearchEngine()
    
    # Test identical vectors
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([1, 2, 3])
    similarity = engine.cosine_similarity(vec1, vec2)
    assert abs(similarity - 1.0) < 0.0001, "Identical vectors should have similarity 1.0"
    print("  ✓ Identical vectors: similarity = 1.0")
    
    # Test orthogonal vectors
    vec3 = np.array([1, 0, 0])
    vec4 = np.array([0, 1, 0])
    similarity = engine.cosine_similarity(vec3, vec4)
    assert abs(similarity - 0.0) < 0.0001, "Orthogonal vectors should have similarity 0.0"
    print("  ✓ Orthogonal vectors: similarity = 0.0")
    
    # Test opposite vectors
    vec5 = np.array([1, 2, 3])
    vec6 = np.array([-1, -2, -3])
    similarity = engine.cosine_similarity(vec5, vec6)
    assert abs(similarity - (-1.0)) < 0.0001, "Opposite vectors should have similarity -1.0"
    print("  ✓ Opposite vectors: similarity = -1.0")
    
    print("✓ Cosine similarity tests passed!\n")


def test_search_engine():
    """Test full search engine."""
    print("Testing search engine...")
    
    # Load documents
    loader = DocumentLoader('data/raw_texts')
    documents = loader.load_documents()
    
    # Index documents
    engine = SearchEngine()
    engine.index_documents(documents)
    
    # Test queries
    test_queries = [
        ("detective mystery", "Should find Sherlock Holmes"),
        ("whale ocean", "Should find Moby Dick"),
        ("vampire blood", "Should find Dracula"),
        ("wonderland rabbit", "Should find Alice in Wonderland"),
    ]
    
    print("\nRunning test queries:")
    print("-"*70)
    
    for query, expected in test_queries:
        print(f"\nQuery: '{query}'")
        print(f"Expected: {expected}")
        
        results = engine.search(query, top_k=3)
        
        if results:
            print(f"Top result: {results[0]['title']} (score: {results[0]['score']:.4f})")
            print("✓ Got results")
        else:
            print("✗ No results found")
    
    print("\n" + "="*70)
    print("✓ Search engine tests completed!\n")


def test_edge_cases():
    """Test edge cases."""
    print("Testing edge cases...")
    
    loader = DocumentLoader('data/raw_texts')
    documents = loader.load_documents()
    
    engine = SearchEngine()
    engine.index_documents(documents)
    
    # Empty query
    print("  Testing empty query...")
    results = engine.search("", top_k=5)
    print(f"    Empty query returned {len(results)} results")
    
    # Single character query
    print("  Testing single character query...")
    results = engine.search("a", top_k=5)
    print(f"    Single char query returned {len(results)} results")
    
    # Very long query
    print("  Testing long query...")
    long_query = "detective mystery adventure journey love romance ocean whale monster creature " * 10
    results = engine.search(long_query, top_k=5)
    print(f"    Long query returned {len(results)} results")
    
    # Special characters
    print("  Testing special characters...")
    results = engine.search("!@#$%^&*()", top_k=5)
    print(f"    Special chars query returned {len(results)} results")
    
    print("✓ Edge case tests completed!\n")


def main():
    print("="*70)
    print("SEARCH ENGINE TEST SUITE")
    print("="*70)
    print()
    
    test_cosine_similarity()
    test_search_engine()
    test_edge_cases()
    
    print("="*70)
    print("ALL TESTS COMPLETED!")
    print("="*70)


if __name__ == '__main__':
    main()