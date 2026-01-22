import os
"""
def load_documents(path):
    #Loads raw documents from disk.
    documents = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), encoding="utf-8") as f:
                documents.append(f.read())
    return documents
"""

"""
Document loader module for text search engine.
Loads and manages text documents from local files.
"""

import os
from typing import List, Dict


class DocumentLoader:
    """Loads text documents from a directory."""
    
    def __init__(self, data_dir: str):
        """
        Initialize loader with data directory.
        
        Args:
            data_dir: Path to directory containing text files
        """
        self.data_dir = data_dir
        self.documents: List[Dict[str, str]] = []
    
    def load_documents(self, encoding: str = 'utf-8') -> List[Dict[str, str]]:
        """
        Load all .txt files from data directory.
        
        Args:
            encoding: Text file encoding (default: utf-8)
            
        Returns:
            List of document dictionaries with 'title', 'content', 'filepath'
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        documents = []
        
        # Get all .txt files
        txt_files = [f for f in os.listdir(self.data_dir) if f.endswith('.txt')]
        
        if not txt_files:
            raise ValueError(f"No .txt files found in {self.data_dir}")
        
        print(f"Found {len(txt_files)} text files")
        
        # Load each file
        for filename in txt_files:
            filepath = os.path.join(self.data_dir, filename)
            
            try:
                with open(filepath, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()
                
                # Create document dict
                doc = {
                    'title': filename.replace('.txt', '').replace('_', ' ').title(),
                    'content': content,
                    'filepath': filepath
                }
                
                documents.append(doc)
                print(f"  Loaded: {doc['title']} ({len(content)} chars)")
                
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
                continue
        
        self.documents = documents
        return documents
    
    def get_document_by_title(self, title: str) -> Dict[str, str]:
        """
        Retrieve a specific document by title.
        
        Args:
            title: Document title to search for
            
        Returns:
            Document dictionary or None if not found
        """
        for doc in self.documents:
            if doc['title'].lower() == title.lower():
                return doc
        return None
    
    def get_document_count(self) -> int:
        """Return number of loaded documents."""
        return len(self.documents)