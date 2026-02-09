"""
Test evaluation metrics.
"""

import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.loader import DocumentLoader
from src.search import SearchEngine
from src.evaluation import SearchEvaluator


def main():
    print("="*70)
    print("EVALUATION METRICS TEST")
    print("="*70)
    print()
    
    # Load and index documents
    loader = DocumentLoader('data/raw_texts')
    documents = loader.load_documents()
    
    engine = SearchEngine()
    engine.index_documents(documents)
    
    evaluator = SearchEvaluator()
    
    # Define test cases with known relevant documents
    # You'll need to manually identify which docs are relevant for each query
    test_cases = [
        {
            'query': 'detective mystery crime',
            'relevant_docs': {2},  # Adjust based on your doc order (e.g., Sherlock Holmes)
            'description': 'Detective/Mystery query'
        },
        {
            'query': 'whale ocean sea',
            'relevant_docs': {3},  # Adjust based on your doc order (e.g., Moby Dick)
            'description': 'Ocean/Whale query'
        },
        {
            'query': 'vampire blood night',
            'relevant_docs': {1},  # Adjust based on your doc order (e.g., Dracula)
            'description': 'Vampire query'
        }
    ]
    
    print("Running evaluation on test queries:")
    print("-"*70)
    
    for test in test_cases:
        query = test['query']
        relevant_docs = test['relevant_docs']
        
        # Search
        results = engine.search(query, top_k=5)
        retrieved_docs = [r['doc_index'] for r in results]
        
        # Calculate metrics
        p_at_3 = evaluator.precision_at_k(relevant_docs, retrieved_docs, 3)
        p_at_5 = evaluator.precision_at_k(relevant_docs, retrieved_docs, 5)
        recall_at_5 = evaluator.recall_at_k(relevant_docs, retrieved_docs, 5)
        avg_prec = evaluator.average_precision(relevant_docs, retrieved_docs)
        
        print(f"\n{test['description']}: '{query}'")
        print(f"  Precision@3: {p_at_3:.3f}")
        print(f"  Precision@5: {p_at_5:.3f}")
        print(f"  Recall@5: {recall_at_5:.3f}")
        print(f"  Average Precision: {avg_prec:.3f}")
        
        print(f"\n  Top 3 results:")
        for i, result in enumerate(results[:3], 1):
            marker = "✓" if result['doc_index'] in relevant_docs else " "
            print(f"    {marker} {i}. {result['title']} (score: {result['score']:.4f})")
    
    print("\n" + "="*70)
    print("✓ Evaluation completed!")
    print("="*70)


if __name__ == '__main__':
    main()