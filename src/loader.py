import os

def load_documents(path):
    """Loads raw documents from disk."""
    documents = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), encoding="utf-8") as f:
                documents.append(f.read())
    return documents
