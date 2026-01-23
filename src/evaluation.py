"""
Evaluation metrics for search engine.
"""

from typing import List, Set


class SearchEvaluator:
    """Evaluate search engine performance."""
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def precision_at_k(self, relevant_docs: Set[int], retrieved_docs: List[int], k: int) -> float:
        """
        Calculate Precision@K.
        
        Args:
            relevant_docs: Set of relevant document indices
            retrieved_docs: List of retrieved document indices (ranked)
            k: Number of top results to consider
            
        Returns:
            Precision@K score (0 to 1)
        """
        if k <= 0 or not retrieved_docs:
            return 0.0
        
        # Get top K retrieved docs
        top_k = retrieved_docs[:k]
        
        # Count how many are relevant
        relevant_retrieved = sum(1 for doc in top_k if doc in relevant_docs)
        
        return relevant_retrieved / k
    
    def recall_at_k(self, relevant_docs: Set[int], retrieved_docs: List[int], k: int) -> float:
        """
        Calculate Recall@K.
        
        Args:
            relevant_docs: Set of relevant document indices
            retrieved_docs: List of retrieved document indices (ranked)
            k: Number of top results to consider
            
        Returns:
            Recall@K score (0 to 1)
        """
        if not relevant_docs or not retrieved_docs:
            return 0.0
        
        # Get top K retrieved docs
        top_k = retrieved_docs[:k]
        
        # Count how many relevant docs were retrieved
        relevant_retrieved = sum(1 for doc in top_k if doc in relevant_docs)
        
        return relevant_retrieved / len(relevant_docs)
    
    def average_precision(self, relevant_docs: Set[int], retrieved_docs: List[int]) -> float:
        """
        Calculate Average Precision.
        
        Args:
            relevant_docs: Set of relevant document indices
            retrieved_docs: List of retrieved document indices (ranked)
            
        Returns:
            Average Precision score (0 to 1)
        """
        if not relevant_docs or not retrieved_docs:
            return 0.0
        
        precisions = []
        relevant_count = 0
        
        for i, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_docs:
                relevant_count += 1
                precision = relevant_count / i
                precisions.append(precision)
        
        if not precisions:
            return 0.0
        
        return sum(precisions) / len(relevant_docs)