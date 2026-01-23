"""
Search engine module with cosine similarity ranking.
"""

import numpy as np
from typing import List, Dict, Tuple
from src.preprocessing import TextPreprocessor
from src.vectorizer import TFIDFVectorizer


class SearchEngine:
    """TF-IDF based document search engine."""
    
    def __init__(self):
        """Initialize search engine components."""
        self.preprocessor = TextPreprocessor(use_stemming=True, remove_stopwords=True)
        self.vectorizer = TFIDFVectorizer()
        self.documents = []
        self.doc_vectors = None
        self.is_fitted = False
    
    def index_documents(self, documents: List[Dict[str, str]]) -> None:
        """
        Build search index from documents.
        
        Args:
            documents: List of document dicts with 'title' and 'content'
        """
        print("Indexing documents...")
        self.documents = documents
        
        # Preprocess all documents
        print("  Preprocessing text...")
        contents = [doc['content'] for doc in documents]
        processed_docs = self.preprocessor.preprocess_documents(contents)
        
        # Build TF-IDF vectors
        print("  Building TF-IDF vectors...")
        self.doc_vectors = self.vectorizer.fit_transform(processed_docs)
        
        self.is_fitted = True
        print(f"✓ Indexed {len(documents)} documents")
        print(f"✓ Vocabulary size: {len(self.vectorizer.vocabulary)}")
        print()
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        # Handle zero vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, any]]:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of result dicts with 'rank', 'title', 'score', 'preview'
        """
        if not self.is_fitted:
            raise ValueError("Search engine not fitted. Call index_documents() first.")
        
        # Preprocess query
        query_tokens = self.preprocessor.preprocess(query)
        
        if not query_tokens:
            print("Warning: Query resulted in no tokens after preprocessing")
            return []
        
        # Convert query to TF-IDF vector
        query_vector = self.vectorizer.transform(query_tokens)
        
        # Calculate similarities with all documents
        similarities = []
        for i, doc_vector in enumerate(self.doc_vectors):
            score = self.cosine_similarity(query_vector, doc_vector)
            similarities.append((i, score))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top K results
        results = []
        for rank, (doc_idx, score) in enumerate(similarities[:top_k], 1):
            if score > 0:  # Only return documents with non-zero similarity
                doc = self.documents[doc_idx]
                
                # Create preview (first 200 chars)
                preview = doc['content'][:200].strip()
                if len(doc['content']) > 200:
                    preview += "..."
                
                results.append({
                    'rank': rank,
                    'title': doc['title'],
                    'score': score,
                    'preview': preview,
                    'doc_index': doc_idx
                })
        
        return results
    
    def print_results(self, query: str, results: List[Dict[str, any]]) -> None:
        """
        Pretty print search results.
        
        Args:
            query: Original search query
            results: List of search results
        """
        print(f"Search query: '{query}'")
        print(f"Found {len(results)} relevant documents")
        print("="*70)
        print()
        
        if not results:
            print("No results found.")
            return
        
        for result in results:
            print(f"Rank {result['rank']}: {result['title']}")
            print(f"Score: {result['score']:.4f}")
            print(f"Preview: {result['preview']}")
            print("-"*70)
            print()