# Vector-Based Text Search Engine

A from-scratch implementation of a document search engine using TF-IDF vectorisation and cosine similarity. Built entirely in Python programming language with no pre-built search libraries.

## Project Overview

This search engine indexes classic literature and enables intelligent searches across multiple documents. The system preprocesses text, builds TF-IDF representations, and ranks documents using cosine similarity to deliver relevant results efficiently.

**Key Features:**
- Custom TF-IDF vectoriser implementation from scratch
- Configurable text preprocessing (stemming, stopword removal)
- Cosine similarity ranking for document retrieval
- Evaluation metrics (Precision@K, Recall@K, Average Precision)
- Interactive search interface

## Technologies Used

- **Python 3.9+**
- **NumPy** - Numerical computations and vector operations
- **Pandas** - Data manipulation
- **NLTK** - Tokenisation and linguistic preprocessing
- **Scikit-learn** - Validation and comparison benchmarks
- **Matplotlib** - Evaluation visualizations

## Project Structure
```
python-text-search-engine/
├── src/
│   ├── loader.py           # Document loading and management
│   ├── preprocessing.py    # Text cleaning and tokenisation
│   ├── vectorizer.py       # TF-IDF implementation
│   ├── search.py           # Search engine with cosine similarity
│   └── evaluation.py       # Performance metrics
├── data/
│   └── raw_texts/          # Classic literature corpus
├── demo_search.py          # Interactive search interface
├── test_*.py               # Unit tests for each module
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Windows/Linux/macOS compatible

### Installation

1. **Clone the repository:**
```bash
   git clone https://github.com/ConstantlyTrying989/python-text-search-engine.git
   cd python-text-search-engine
```

2. **Create virtual environment:**
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate.bat
```

3. **Install dependencies:**
```bash
   pip install -r requirements.txt
```

4. **Download NLTK data (for the first run):**
```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Quick Start

**Run the interactive search demo:**
```bash
python demo_search.py
```

**Example queries to try:**
- `detective mystery crime`
- `ocean whale adventure`
- `love romance marriage`

## How It Works

### 1. Document Loading
The system loads plain text documents, indexing a collection of classic novels.

### 2. Text Preprocessing
- Converts text to lowercase
- Removes URLs, emails, and numbers
- Tokenizes using NLTK's word_tokenize
- Removes stopwords (common words like "and")
- Applies Porter stemming to reduce words to root forms

### 3. TF-IDF Vectorization
Implements Term Frequency-Inverse Document Frequency from scratch:
- **TF (Term Frequency):** Normalized word frequency in document
- **IDF (Inverse Document Frequency):** log(N / document_frequency)
- Creates sparse vector representation for each document

### 4. Cosine Similarity Ranking
Compares query vector to document vectors using cosine similarity:
```
similarity = (A · B) / (||A|| × ||B||)
```
Returns top K most similar documents ranked by score.

### 5. Evaluation Metrics
- **Precision@K:** Accuracy of top K results
- **Recall@K:** Coverage of relevant documents in top K
- **Average Precision:** Overall ranking quality

## Testing

**Run all tests:**
```bash
python test_loader.py
python test_preprocessing.py
python test_vectorizer.py
python test_search.py
python test_evaluation.py
```

**Expected output:**
- Document loading verification
- Preprocessing statistics
- TF-IDF matrix dimensions
- Search result rankings
- Evaluation metric scores

## Performance

The search engine efficiently processes queries across the indexed corpus:
- Optimized TF-IDF computation
- Quick document retrieval and ranking
- Sub-second average query time
- Scalable vector operations using NumPy

**Sample Results:**
Query-dependent results ranked by cosine similarity scores.

## Technical Highlights

**Custom Implementation:**
- No use of sklearn's TfidfVectorizer
- Manual cosine similarity computation
- Custom evaluation metrics for testing

**Algorithmic Complexity:**
- Indexing: O(N × M) where N = documents, M = avg tokens
- Query: O(V) where V = vocabulary size
- Memory: O(N × V) for document-term matrix

## Learning Outcomes

This project demonstrates:
- Information retrieval fundamentals (TF-IDF, cosine similarity)
- Text preprocessing and NLP pipeline design
- NumPy for efficient vector operations
- Software engineering best practices (testing)
- Working with real-world text data

## Future Enhancements

- [ ] Implement search for synonyms in the texts
- [ ] Add phrase search support
- [ ] Add relevance feedback mechanism

## Contributing

This is a portfolio project, but suggestions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is open source and available under the MIT License.

## Author

**Jason Lewis**
- GitHub: [@ConstantlyTrying989](https://github.com/ConstantlyTrying989)
- LinkedIn: [Jason Lewis](https://linkedin.com/in/jason-lewis-b3a188288)
- Email: lewisjd2007@gmail.com

## Acknowledgments

- Project Gutenberg for public domain texts
- NLTK team for NLP tools
- Classic literature authors for the corpus

---

