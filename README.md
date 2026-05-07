# 🏥 Medical NLP Pipeline: PubMed Literature Analysis

A complete Python data science project for analyzing biomedical literature from PubMed using natural language processing and machine learning. This pipeline integrates multiple NLP techniques to extract knowledge and detect healthcare trends from medical abstracts.

## 📋 Project Overview

This project provides an end-to-end solution for:
- **Fetching** biomedical literature from PubMed using the NCBI Entrez API
- **Preprocessing** medical abstracts with NLTK and spaCy
- **Analyzing** document characteristics and trends (EDA)
- **Extracting** medical named entities using spaCy NER
- **Modeling** latent topics with Gensim LDA
- **Computing** semantic similarity with Word2Vec
- **Classifying** abstracts into medical specialties using scikit-learn

All components are wrapped in an interactive Streamlit web application and containerized with Docker.

## ✨ Features

### 1. Data Collection
- ✅ Fetch abstracts from PubMed via Biopython Entrez API
- ✅ Rate limiting and retry logic for reliable data collection
- ✅ Extract: PMID, Title, Abstract, Publication Date, Journal, Keywords
- ✅ Save to CSV format

### 2. Exploratory Data Analysis
- ✅ Dataset statistics and missing value analysis
- ✅ Abstract length distribution
- ✅ Publication trends by year
- ✅ Most frequent terms
- ✅ Word clouds
- ✅ Interactive visualizations

### 3. Text Preprocessing
- ✅ HTML/URL/citation removal
- ✅ Tokenization with NLTK
- ✅ Custom academic stopword removal
- ✅ lemmatization with spaCy
- ✅ POS-tag filtering (NOUN, PROPN, ADJ)
- ✅ Batch processing for efficiency

### 4. Named Entity Recognition
- ✅ Extract medical entities using spaCy
- ✅ Fallback to standard model if biomedical model unavailable
- ✅ Entity frequency analysis
- ✅ Entity label distribution
- ✅ Visualizations and statistics

### 5. Topic Modeling
- ✅ Gensim LDA implementation
- ✅ Configurable number of topics (2-10)
- ✅ Coherence score calculation
- ✅ Topic-document distributions
- ✅ Top words per topic visualization

### 6. Semantic Similarity
- ✅ Word2Vec model training
- ✅ Find similar medical terms
- ✅ Word analogy support
- ✅ Vocabulary browser

### 7. Medical Specialty Classification
- ✅ Auto-label abstracts (Oncology, Cardiology, Infectious Disease, Neurology, Other)
- ✅ Random Forest and SVM classifiers
- ✅ TF-IDF feature extraction
- ✅ Confusion matrix, F1-score, classification reports
- ✅ Per-sample predictions with confidence scores

### 8. Interactive Web Interface
- ✅ 7-page Streamlit application
- ✅ Real-time data processing
- ✅ Caching for performance
- ✅ Download results as CSV
- ✅ Responsive design

### 9. Docker Containerization
- ✅ Docker and Docker Compose support
- ✅ Single command deployment: `docker compose up --build`
- ✅ Volume mounts for data persistence
- ✅ Pre-installed spaCy and NLTK models

## 🚀 Quick Start

### Option 1: Docker (Recommended)

**Requirements:**
- Docker and Docker Compose installed
- ~5-10 minutes for first-time build (downloads dependencies)

**Steps:**

```bash
# Navigate to project directory
cd pubmed-medical-nlp

# Build and run
docker compose up --build

# Open browser
# http://localhost:8501
```

The app will be available at `http://localhost:8501` after startup (~30-60 seconds).

**Stop the app:**
```bash
docker compose down
```

### Option 2: Local Installation (Python 3.11+)

**Requirements:**
- Python 3.11 or higher
- pip package manager
- ~15 minutes setup time

**Steps:**

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -m nltk.downloader punkt stopwords

# Set Entrez email (required for PubMed API)
# On Windows (PowerShell):
$env:ENTREZ_EMAIL="your-email@example.com"
# On macOS/Linux:
export ENTREZ_EMAIL="your-email@example.com"

# Run Streamlit app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## 📖 Usage Guide

### Page 1: Dataset & PubMed Search
1. Enter a medical query (e.g., "cancer immunotherapy", "COVID-19 vaccine")
2. Choose number of abstracts (10-10,000)
3. Click "Fetch PubMed Data"
4. View sample data and summary statistics
5. Optionally download raw CSV

**Default Query:** "cancer immunotherapy" (100 abstracts)

### Page 2: Exploratory Data Analysis
- View dataset statistics
- Analyze abstract length distribution
- Track publication trends by year
- Identify most frequent terms
- Generate word clouds
- Visualize missing values

### Page 3: Text Preprocessing
1. Click "Process All Abstracts"
2. View before/after comparison
3. Check preprocessing statistics
4. Download processed dataset

**Processing includes:**
- HTML/URL/citation removal
- Tokenization
- Stopword removal
- Lemmatization
- POS filtering

### Page 4: Named Entity Recognition
1. Click "Extract Named Entities"
2. Explore entity distribution by label
3. View top named entities
4. See sample entities from random abstract
5. Access entity statistics

### Page 5: Topic Modeling
1. Choose number of topics (2-10)
2. Click "Train LDA Model"
3. View top words per topic
4. Check average topic distribution
5. Query topic distribution for specific documents

**Coherence Score:** Higher is better (0.0-1.0 range)

### Page 6: Word2Vec Semantic Similarity
1. Click "Train Word2Vec Model"
2. Enter a medical term (e.g., "cancer", "vaccine", "diabetes")
3. Set number of similar words (5-20)
4. Click "Find Similar Words"
5. Browse complete vocabulary

**Example queries:**
- "cancer" → oncology, tumor, carcinoma, ...
- "vaccine" → immunization, vaccination, antibody, ...
- "heart" → cardiac, cardiovascular, hypertension, ...

### Page 7: ML Classification
1. Choose classifier type (Random Forest or SVM)
2. Click "Train Classifier"
3. Review performance metrics
4. View confusion matrix
5. Make predictions on sample abstracts

**Medical Specialties:**
- **Oncology** - Cancer, tumors, chemotherapy
- **Cardiology** - Heart, cardiac, hypertension
- **Infectious Disease** - Infection, virus, vaccine, COVID
- **Neurology** - Brain, neurological, Alzheimer's, stroke
- **Other** - All other topics

## 📁 Project Structure

```
pubmed-medical-nlp/
│
├── app.py                          # Main Streamlit application (7 pages)
│
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── pubmed_api.py               # PubMed data fetching (Biopython Entrez)
│   ├── preprocessing.py            # Text preprocessing (NLTK + spaCy)
│   ├── eda.py                      # Exploratory data analysis
│   ├── ner.py                      # Named entity recognition (spaCy)
│   ├── topic_modeling.py           # LDA topic modeling (Gensim)
│   ├── word2vec_model.py           # Word2Vec semantic similarity (Gensim)
│   └── classifier.py               # Medical specialty classification (scikit-learn)
│
├── data/
│   ├── raw/                        # Raw PubMed CSVs (auto-created)
│   │   └── pubmed_data.csv
│   └── processed/                  # Cleaned abstracts (auto-created)
│       └── pubmed_processed.csv
│
├── outputs/
│   ├── figures/                    # Generated visualizations (auto-created)
│   │   ├── abstract_length_dist.png
│   │   ├── publication_trend.png
│   │   └── ...
│   └── models/                     # Trained models (auto-created)
│       ├── lda_model.model
│       ├── word2vec_model.model
│       ├── classifier.pkl
│       └── tfidf.pkl
│
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container image
├── docker-compose.yml              # Docker Compose config
├── README.md                       # This file
└── .gitignore                      # Git ignore rules
```

## 📦 Dependencies

### Core Libraries
- `streamlit` - Interactive web framework
- `pandas` - Data manipulation
- `numpy` - Numerical computing

### NLP/Text Processing
- `nltk` - Natural Language Toolkit
- `spacy` - Industrial-strength NLP
- `gensim` - Topic modeling (LDA) and Word2Vec
- `biopython` - PubMed API access

### Machine Learning
- `scikit-learn` - ML algorithms and evaluation

### Visualization
- `matplotlib` - Static plots
- `seaborn` - Statistical visualization
- `wordcloud` - Word cloud generation
- `plotly` - Interactive plots (optional)

### Utilities
- `python-dotenv` - Environment variables
- `tqdm` - Progress bars

All dependencies are listed in `requirements.txt`.

## 🔧 Configuration

### Environment Variables

**Required:**
```bash
ENTREZ_EMAIL=your-email@example.com
```

This is required by NCBI for responsible API usage. You can set it in:
- Docker Compose: Edit `docker-compose.yml`
- Local: `export ENTREZ_EMAIL="..."` (macOS/Linux) or PowerShell command
- .env file: Create `.env` with `ENTREZ_EMAIL=...`

**Optional:**
```bash
STREAMLIT_SERVER_PORT=8501        # Default port
STREAMLIT_SERVER_ADDRESS=0.0.0.0  # Default address
```

### PubMed API Rate Limiting

The PubMed API has rate limits:
- **3 requests/second** without API key
- **10 requests/second** with API key

The code implements:
- 0.35 second delay between requests
- Exponential backoff on errors
- Batch fetching to reduce requests

For large-scale projects (>100K abstracts), consider getting an NCBI API key.

## 📊 Sample Results

### With Query: "cancer immunotherapy"
- **Abstracts Retrieved:** 100+
- **Average Abstract Length:** 1500-2000 characters
- **Publication Years:** 2010-2025
- **Entity Labels Found:** PERSON, ORG, GPE, EVENT (and custom NER)
- **Top Topics:** Immunotherapy mechanisms, Cancer types, Treatment outcomes, Clinical trials
- **Most Similar to "cancer":** oncology, tumor, carcinoma, malignancy, neoplasm

## 🐛 Troubleshooting

### Docker Issues

**Problem:** "Error response from daemon: Bind for 0.0.0.0:8501 failed"
- **Solution:** Port 8501 is in use. Change in `docker-compose.yml`: `"8502:8501"`

**Problem:** "No space left on device"
- **Solution:** Docker images/containers taking space. Run `docker system prune`

**Problem:** "ModuleNotFoundError: No module named spacy"
- **Solution:** Wait for Docker build to complete, or rebuild: `docker compose up --build`

### Local Installation Issues

**Problem:** "No module named spacy"
```bash
# Install missing packages
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords
```

**Problem:** "ENTREZ_EMAIL not set"
```bash
export ENTREZ_EMAIL="your-email@example.com"
streamlit run app.py
```

**Problem:** "ModuleNotFoundError: No module named src"
```bash
# Ensure you're in the project directory
cd pubmed-medical-nlp
streamlit run app.py
```

### PubMed API Issues

**Problem:** "Error fetching data: 429 Too Many Requests"
- **Solution:** API rate limit exceeded. Wait a minute and retry. Implement longer delays if needed.

**Problem:** "Error fetching data: Connection timeout"
- **Solution:** Network issue. Check internet connection and retry.

**Problem:** "No results found for query"
- **Solution:** Try a different, more specific query. Example: "cancer treatment 2020-2025"

### Memory Issues

**Problem:** "MemoryError" with >100K abstracts
- **Solutions:**
  1. Reduce number of abstracts in slider
  2. Increase available RAM or run on machine with more memory
  3. Process data in batches in code (advanced)

**Problem:** Word2Vec/LDA models slow
- **Solution:** These models are computation-intensive. Training on 10K+ abstracts may take 5-10 minutes. Be patient!

## 🔬 Validation & Testing

### Quick Test (2-3 minutes)
1. Run with default query "cancer immunotherapy" + 100 abstracts
2. Complete entire pipeline from Dataset → Classification
3. Check all visualizations render correctly

### Full Test (10-15 minutes)
1. Fetch 1000 abstracts
2. Run all analyses
3. Train all models
4. Verify predictions make sense (e.g., "cancer" documents → Oncology)

### Validation Checks

- ✅ **Data Collection:** CSV has all 6 columns (PMID, Title, Abstract, PublicationDate, Journal, Keywords)
- ✅ **Preprocessing:** Token reduction 30-50%, lemmatization working
- ✅ **NER:** Entities extracted, distributed across multiple labels
- ✅ **Topic Modeling:** Coherence >0.4, topics readable
- ✅ **Word2Vec:** Query "cancer" returns medical terms (similarity 0.5-0.9)
- ✅ **Classification:** Accuracy >0.5, confusion matrix shows reasonable predictions
- ✅ **Streamlit App:** All 7 pages load, no errors

## 📈 Advanced Usage

### Custom Keyword Labels (Classifier)

Edit `src/classifier.py` to add more specialties:

```python
SPECIALTY_KEYWORDS = {
    'Your Specialty': {'keyword1', 'keyword2', ...},
    ...
}
```

### Biomedical spaCy Model

For better medical NER, install biomedical model:

```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz
```

Then in app.py, change to:
```python
st.session_state.ner_extractor = EntityExtractor(model_name='en_core_sci_md')
```

### Save Trained Models

Models are auto-saved to `outputs/models/`. To load them later:

```python
from gensim.models import LdaModel, Word2Vec
import pickle

# Load LDA
lda = LdaModel.load('outputs/models/lda_model.model')

# Load Word2Vec
w2v = Word2Vec.load('outputs/models/word2vec_model.model')

# Load Classifier
with open('outputs/models/classifier.pkl', 'rb') as f:
    clf = pickle.load(f)
```

### Batch Processing

For processing large numbers of abstracts, modify `app.py` to add batch job processing.

## 📚 References

- **PubMed API**: https://www.ncbi.nlm.nih.gov/books/NBK25497/
- **Biopython Entrez**: https://biopython.org/wiki/Documentation
- **NLTK Documentation**: https://www.nltk.org/
- **spaCy Models**: https://spacy.io/models
- **Gensim LDA**: https://radimrehurek.com/gensim/models/ldamodel.html
- **Word2Vec**: https://radimrehurek.com/gensim/models/word2vec.html
- **Streamlit Documentation**: https://docs.streamlit.io/

## ✅ Code Quality

- ✅ Modular design: Each component in separate file
- ✅ Comprehensive docstrings and comments
- ✅ Error handling and logging throughout
- ✅ Type hints for better IDE support
- ✅ Efficient batch processing with spaCy
- ✅ Streamlit caching for performance
- ✅ No hardcoded paths (uses `pathlib`)

## 🎯 Future Enhancements

- [ ] Sentiment analysis layer
- [ ] Interactive entity visualization (spaCy displacy)
- [ ] PDF report export
- [ ] Document similarity search (LSA/LDA-based)
- [ ] REST API endpoint
- [ ] Database storage (PostgreSQL)
- [ ] Advanced hyperparameter tuning UI
- [ ] Comparison with multiple models
- [ ] Real-time streaming updates
- [ ] Multi-language support

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

## 💬 Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review error messages in Streamlit terminal
3. Check PubMed API documentation
4. Verify environment setup (ENTREZ_EMAIL, Python version)

## 📞 Contact & Attribution

**Author:** Medical NLP Pipeline Project

**Acknowledgments:**
- NCBI PubMed for literature data
- Biopython community
- spaCy team for NLP models
- Gensim developers
- Streamlit team

---

**Last Updated:** April 2026  
**Python Version:** 3.11+  
**Status:** Production Ready ✅
#   n l p _ p r o j e c t  
 