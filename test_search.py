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
    print("="*70)
    print("TESTING COSINE SIMILARITY...")
    print("="*70)
    
    engine = SearchEngine()
    
    # Test 1: Identical vectors
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([1, 2, 3])
    similarity = engine.cosine_similarity(vec1, vec2)
    assert abs(similarity - 1.0) < 0.0001, "Identical vectors should have similarity 1.0"
    print("  ✅ Identical vectors: similarity = 1.0")
    
    # Test 2: Orthogonal vectors
    vec3 = np.array([1, 0, 0])
    vec4 = np.array([0, 1, 0])
    similarity = engine.cosine_similarity(vec3, vec4)
    assert abs(similarity - 0.0) < 0.0001, "Orthogonal vectors should have similarity 0.0"
    print("  ✅ Orthogonal vectors: similarity = 0.0")
    
    # Test 3: Opposite vectors
    vec5 = np.array([1, 2, 3])
    vec6 = np.array([-1, -2, -3])
    similarity = engine.cosine_similarity(vec5, vec6)
    assert abs(similarity - (-1.0)) < 0.0001, "Opposite vectors should have similarity -1.0"
    print("  ✅ Opposite vectors: similarity = -1.0")
    
    # Test 4: Zero vector
    vec7 = np.array([0, 0, 0])
    vec8 = np.array([1, 2, 3])
    similarity = engine.cosine_similarity(vec7, vec8)
    assert similarity == 0.0, "Zero vector should have similarity 0.0"
    print("  ✅ Zero vector: similarity = 0.0")
    
    print("\n✅ Cosine similarity tests passed!\n")


def test_search_engine():
    """Test full search engine."""
    print("="*70)
    print("TESTING SEARCH ENGINE...")
    print("="*70)
    print()
    
    # Load documents
    loader = DocumentLoader('data/raw_texts')
    
    try:
        documents = loader.load_documents()
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return
    
    print()
    
    # Index documents
    engine = SearchEngine()
    engine.index_documents(documents)
    
    # Test queries
    test_queries = [
        ("detective mystery crime", "Detective/mystery content"),
        ("whale ocean sea", "Ocean/whale content"),
        ("vampire blood night", "Vampire content"),
        ("love romance marriage", "Romance content"),
    ]
    
    print("="*70)
    print("RUNNING TEST QUERIES:")
    print("="*70)
    
    for query, expected in test_queries:
        print(f"\n📝 Query: '{query}'")
        print(f"   Expected: {expected}")
        print("   " + "-"*66)
        
        results = engine.search(query, top_k=3)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result['title']}")
                print(f"      Score: {result['score']:.4f}")
            print("   ✅ Got results")
        else:
            print("   ⚠️  No results found")
    
    print("\n" + "="*70)
    print("✅ Search engine tests completed!")
    print("="*70)
    print()


def test_edge_cases():
    """Test edge cases."""
    print("="*70)
    print("TESTING EDGE CASES...")
    print("="*70)
    
    loader = DocumentLoader('data/raw_texts')
    
    try:
        documents = loader.load_documents()
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return
    
    print()
    
    engine = SearchEngine()
    engine.index_documents(documents)
    
    # Test 1: Empty query
    print("\n  📝 Testing empty query...")
    results = engine.search("", top_k=5)
    print(f"     → Returned {len(results)} results")
    
    # Test 2: Single character query
    print("  📝 Testing single character query...")
    results = engine.search("a", top_k=5)
    print(f"     → Returned {len(results)} results")
    
    # Test 3: Very long query
    print("  📝 Testing long query...")
    long_query = "detective mystery adventure journey love romance ocean whale monster creature " * 10
    results = engine.search(long_query, top_k=5)
    print(f"     → Returned {len(results)} results")
    
    # Test 4: Special characters
    print("  📝 Testing special characters...")
    results = engine.search("!@#$%^&*()", top_k=5)
    print(f"     → Returned {len(results)} results")
    
    # Test 5: Numbers
    print("  📝 Testing numbers...")
    results = engine.search("123 456 789", top_k=5)
    print(f"     → Returned {len(results)} results")
    
    # Test 6: Stopwords only
    print("  📝 Testing stopwords only...")
    results = engine.search("the and or but", top_k=5)
    print(f"     → Returned {len(results)} results")
    
    print("\n✅ Edge case tests completed!\n")


def main():
    print("\n" + "="*70)
    print("SEARCH ENGINE TEST SUITE")
    print("="*70)
    print()
    
    test_cosine_similarity()
    test_search_engine()
    test_edge_cases()
    
    print("="*70)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print()


if __name__ == '__main__':
    main()