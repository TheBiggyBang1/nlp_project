# 🏥 Medical NLP Pipeline: PubMed Literature Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Ready-0db7ed?style=for-the-badge&logo=docker)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit)

**A comprehensive end-to-end NLP pipeline for biomedical literature analysis from PubMed**

**Fetch • Preprocess • Analyze • Extract • Model • Classify • Visualize**

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-usage-guide) • [📁 Structure](#-project-structure) • [🐛 Help](#-troubleshooting)

</div>

---

## 📚 Overview

An enterprise-grade **Python data science project** for analyzing biomedical literature from PubMed using cutting-edge NLP and machine learning techniques. This pipeline integrates multiple state-of-the-art tools to extract knowledge and detect healthcare trends from medical abstracts.

### What It Does

- 📥 **Fetches** biomedical literature from PubMed using NCBI Entrez API
- 🧹 **Preprocesses** medical abstracts with NLTK and spaCy
- 📊 **Analyzes** document characteristics and trends with EDA
- 🏷️ **Extracts** medical named entities using spaCy NER
- 📚 **Models** latent topics with Gensim LDA
- 🔗 **Computes** semantic similarity with Word2Vec
- 🤖 **Classifies** abstracts into medical specialties using scikit-learn
- 🌐 **Visualizes** results in an interactive Streamlit web application
- 🐳 **Deploys** seamlessly with Docker and Docker Compose

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Advanced Usage](#-advanced-usage)
- [References](#-references)
- [Contributing](#-contributing)

---

## ✨ Features

### 🎯 Core Capabilities

| Feature | Technology | Status |
|---------|-----------|--------|
| **Data Collection** | Biopython Entrez API | ✅ |
| **Text Preprocessing** | NLTK + spaCy | ✅ |
| **Exploratory Analysis** | Matplotlib + Seaborn | ✅ |
| **Named Entity Recognition** | spaCy NER | ✅ |
| **Topic Modeling** | Gensim LDA | ✅ |
| **Semantic Similarity** | Word2Vec | ✅ |
| **Classification** | scikit-learn | ✅ |
| **Web Interface** | Streamlit | ✅ |
| **Containerization** | Docker | ✅ |

### 📊 Data Collection
- ✅ Fetch abstracts from PubMed via Biopython Entrez API
- ✅ Smart rate limiting and exponential backoff retry logic
- ✅ Extract metadata: PMID, Title, Abstract, PublicationDate, Journal, Keywords
- ✅ Automatic CSV export

### 🔍 Exploratory Data Analysis
- ✅ Comprehensive dataset statistics
- ✅ Missing value analysis with visualizations
- ✅ Abstract length distribution plots
- ✅ Publication trends by year
- ✅ Most frequent terms analysis
- ✅ Word cloud generation
- ✅ Interactive visualizations

### 🧹 Text Preprocessing
- ✅ HTML/URL/citation removal
- ✅ NLTK tokenization
- ✅ Custom academic stopword removal
- ✅ spaCy lemmatization
- ✅ POS-tag filtering (NOUN, PROPN, ADJ)
- ✅ Batch processing for efficiency

### 🏷️ Named Entity Recognition
- ✅ Medical entity extraction with spaCy
- ✅ Automatic model fallback
- ✅ Entity frequency analysis
- ✅ Label distribution visualization
- ✅ Statistical summaries

### 📚 Topic Modeling
- ✅ Gensim LDA implementation
- ✅ Configurable topics (2-10)
- ✅ Coherence score calculation
- ✅ Topic-document distribution analysis
- ✅ Top words per topic visualization

### 🔗 Semantic Similarity
- ✅ Word2Vec model training
- ✅ Similar medical terms finder
- ✅ Word analogy support
- ✅ Vocabulary browsing

### 🤖 Medical Classification
- ✅ Auto-label abstracts into specialties:
  - 🔴 **Oncology** - Cancer, tumors, chemotherapy
  - 💙 **Cardiology** - Heart, cardiac, hypertension
  - 🦠 **Infectious Disease** - Infection, virus, vaccine, COVID
  - 🧠 **Neurology** - Brain, neurological, Alzheimer's
  - 📋 **Other** - All other topics
- ✅ Random Forest and SVM classifiers
- ✅ TF-IDF feature extraction
- ✅ Confusion matrix and F1-score metrics
- ✅ Per-sample predictions with confidence

---

## 🚀 Quick Start

### ⚙️ Option 1: Docker (Recommended) 🐳

**Requirements:**
- Docker and Docker Compose
- ~5-10 minutes for first-time setup

**Steps:**

```bash
# Navigate to project directory
cd pubmed-medical-nlp

# Build and run with Docker Compose
docker compose up --build

# Open your browser
# → http://localhost:8501
```

**Stop the application:**
```bash
docker compose down
```

> 💡 **Tip:** First build takes longer due to dependency installation. Subsequent runs will be much faster!

---

### 🐍 Option 2: Local Installation (Python 3.11+)

**Requirements:**
- Python 3.11 or higher
- pip package manager
- ~15 minutes setup time

**Steps:**

```bash
# 1. Navigate to project
cd pubmed-medical-nlp

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download NLP models
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords

# 6. Set environment variable (required for PubMed API)
# Windows PowerShell:
$env:ENTREZ_EMAIL="your-email@example.com"

# macOS/Linux:
export ENTREZ_EMAIL="your-email@example.com"

# 7. Run Streamlit app
streamlit run app.py
```

The app will automatically open at `http://localhost:8501`.

---

## 📖 Usage Guide

### 📄 Page 1: Dataset & PubMed Search
**Fetch and explore biomedical literature**

1. Enter a medical search query (e.g., "cancer immunotherapy", "COVID-19 vaccine")
2. Select number of abstracts to fetch (10-10,000)
3. Click "Fetch PubMed Data"
4. View sample data and statistics
5. Download raw CSV if needed

**Default:** "cancer immunotherapy" with 100 abstracts

---

### 📊 Page 2: Exploratory Data Analysis
**Understand your dataset through visualizations**

- Dataset statistics and summary
- Missing value analysis
- Abstract length distribution
- Publication trends over time
- Most frequent terms
- Word clouds
- Interactive charts

---

### 🧹 Page 3: Text Preprocessing
**Transform raw text into clean tokens**

1. Click "Process All Abstracts"
2. Compare before/after preprocessing
3. Review statistics (tokens removed, reduction %)
4. Download processed dataset

**Pipeline:**
- HTML/URL/citation removal
- Tokenization
- Stopword removal
- Lemmatization
- POS filtering

---

### 🏷️ Page 4: Named Entity Recognition
**Extract medical entities**

1. Click "Extract Named Entities"
2. Explore entity distributions by type
3. View top entities in dataset
4. Analyze entity statistics
5. See sample entities

---

### 📚 Page 5: Topic Modeling
**Discover latent themes with LDA**

1. Select number of topics (2-10)
2. Click "Train LDA Model"
3. View top words per topic
4. Check topic distributions
5. Query topic for specific documents

**Metric:** Coherence Score (0.0-1.0, higher is better)

---

### 🔗 Page 6: Word2Vec Semantic Similarity
**Explore medical term relationships**

1. Click "Train Word2Vec Model"
2. Enter a medical term (e.g., "cancer", "vaccine", "diabetes")
3. Set number of similar words (5-20)
4. View similarity scores
5. Browse vocabulary

**Examples:**
- "cancer" → oncology, tumor, carcinoma, neoplasm...
- "vaccine" → immunization, vaccination, antibody...
- "heart" → cardiac, cardiovascular, hypertension...

---

### 🤖 Page 7: ML Classification
**Classify abstracts into medical specialties**

1. Choose classifier (Random Forest or SVM)
2. Click "Train Classifier"
3. Review performance metrics
4. View confusion matrix
5. Make predictions on samples

---

## 📁 Project Structure

```
pubmed-medical-nlp/
│
├── 📄 app.py                          # Main Streamlit app (7 pages)
│
├── 📁 src/                            # Core NLP modules
│   ├── __init__.py
│   ├── pubmed_api.py                  # 🌐 PubMed fetching
│   ├── preprocessing.py               # 🧹 Text preprocessing
│   ├── eda.py                         # 📊 Data analysis
│   ├── ner.py                         # 🏷️ Entity extraction
│   ├── topic_modeling.py              # 📚 LDA modeling
│   ├── word2vec_model.py              # 🔗 Semantic similarity
│   └── classifier.py                  # 🤖 Classification
│
├── 📁 data/
│   ├── raw/                           # Raw fetched abstracts
│   │   └── pubmed_data.csv
│   └── processed/                     # Cleaned abstracts
│       └── pubmed_processed.csv
│
├── 📁 outputs/
│   ├── figures/                       # Generated visualizations
│   └── models/                        # Trained models
│
├── 📋 requirements.txt                # Python dependencies
├── 🐳 Dockerfile                      # Container image
├── 🐳 docker-compose.yml              # Docker Compose config
├── 📖 README.md                       # This file
└── 🚫 .gitignore                      # Git ignore rules
```

---

## 📦 Dependencies

### Core Libraries
- **streamlit** - Interactive web framework
- **pandas** - Data manipulation
- **numpy** - Numerical computing

### NLP & Text Processing
- **nltk** - Natural Language Toolkit
- **spacy** - Industrial-strength NLP
- **gensim** - Topic modeling & Word2Vec
- **biopython** - PubMed API access

### Machine Learning
- **scikit-learn** - ML algorithms & evaluation

### Visualization
- **matplotlib** - Static plots
- **seaborn** - Statistical visualization
- **wordcloud** - Word cloud generation

### Utilities
- **python-dotenv** - Environment variables
- **tqdm** - Progress bars

See `requirements.txt` for complete list with versions.

---

## 🔧 Configuration

### Environment Variables

**Required:**
```bash
ENTREZ_EMAIL=your-email@example.com  # Required by NCBI for API access
```

**Optional:**
```bash
STREAMLIT_SERVER_PORT=8501           # Custom port
STREAMLIT_SERVER_ADDRESS=0.0.0.0     # Server address
```

### Setting Environment Variables

**Windows PowerShell:**
```powershell
$env:ENTREZ_EMAIL="your-email@example.com"
```

**macOS/Linux:**
```bash
export ENTREZ_EMAIL="your-email@example.com"
```

**Using .env file:**
```bash
# Create .env in project root
echo ENTREZ_EMAIL="your-email@example.com" > .env
```

### PubMed API Rate Limits

| Configuration | Requests/Second |
|---------------|-----------------|
| Without API key | 3 |
| With API key | 10 |

**Our Implementation:**
- 0.35 second delay between requests
- Exponential backoff on errors
- Batch fetching to minimize requests
- Robust error handling

---

## 🐛 Troubleshooting

### 🐳 Docker Issues

**Port already in use:**
```yaml
# Edit docker-compose.yml:
ports:
  - "8502:8501"  # Use different port
```

**No space left on device:**
```bash
docker system prune -a --volumes
docker compose up --build
```

**Module not found errors:**
```bash
docker compose up --build --force-recreate
```

### 💻 Local Installation Issues

**Missing spaCy model:**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords
```

**ENTREZ_EMAIL not set:**
```bash
# Windows:
$env:ENTREZ_EMAIL="your-email@example.com"
streamlit run app.py

# macOS/Linux:
export ENTREZ_EMAIL="your-email@example.com"
streamlit run app.py
```

**Wrong directory error:**
```bash
cd pubmed-medical-nlp
streamlit run app.py
```

### 🌐 PubMed API Issues

**429 Too Many Requests:**
- Wait 1-2 minutes before retrying
- Reduce number of abstracts
- Get an NCBI API key for higher limits

**Connection timeout:**
- Check internet connection
- Verify PubMed is accessible
- Try again after a few seconds

**No results found:**
- Use simpler, broader queries
- Try: "cancer treatment", "COVID-19 vaccine", "diabetes"

### 💾 Performance Issues

**MemoryError with large datasets:**
- Reduce abstracts in slider (try 1,000-5,000)
- Use machine with more RAM (16GB+ for 100K+)

**Models training slowly:**
- This is normal! LDA/Word2Vec are computationally intensive
- Training on 10K+ abstracts takes 5-10 minutes
- Use fewer topics (3-5) for faster results

---

## ✅ Testing & Validation

### Quick Test (2-3 minutes)
1. Default query "cancer immunotherapy" + 100 abstracts
2. Run complete pipeline
3. Verify visualizations render

### Full Test (10-15 minutes)
1. Fetch 1,000 abstracts
2. Run all modules
3. Train all models
4. Verify predictions make sense

### Validation Checklist
- ✅ CSV has all columns (PMID, Title, Abstract, PublicationDate, Journal, Keywords)
- ✅ Preprocessing reduces tokens 30-50%
- ✅ NER extracts entities across multiple labels
- ✅ LDA coherence >0.4
- ✅ Word2Vec similarities 0.5-0.9
- ✅ Classification accuracy >0.5
- ✅ All 7 Streamlit pages load without errors

---

## 📈 Advanced Usage

### Custom Medical Specialties

Edit `src/classifier.py`:

```python
SPECIALTY_KEYWORDS = {
    'Dermatology': {'skin', 'dermatitis', 'eczema', ...},
    'Orthopedics': {'bone', 'joint', 'fracture', ...},
}
```

### Biomedical spaCy Model

```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz
```

Then update `app.py` to use `en_core_sci_md`.

### Load Trained Models

```python
from gensim.models import LdaModel, Word2Vec
import pickle

lda = LdaModel.load('outputs/models/lda_model.model')
w2v = Word2Vec.load('outputs/models/word2vec_model.model')

with open('outputs/models/classifier.pkl', 'rb') as f:
    clf = pickle.load(f)
```

---

## 📚 References

- 🔗 [PubMed API Documentation](https://www.ncbi.nlm.nih.gov/books/NBK25497/)
- 🔗 [Biopython Entrez](https://biopython.org/wiki/Documentation)
- 🔗 [NLTK Documentation](https://www.nltk.org/)
- 🔗 [spaCy Models](https://spacy.io/models)
- 🔗 [Gensim LDA](https://radimrehurek.com/gensim/models/ldamodel.html)
- 🔗 [Streamlit Docs](https://docs.streamlit.io/)

---

## 🎯 Future Enhancements

- [ ] Sentiment analysis layer
- [ ] Interactive entity visualizations
- [ ] PDF report export
- [ ] Document similarity search
- [ ] REST API endpoints
- [ ] Database storage (PostgreSQL)
- [ ] Advanced hyperparameter tuning UI
- [ ] Multi-model comparison
- [ ] Real-time streaming updates
- [ ] Multi-language support

---

## 📝 License

MIT License - Free for research, education, and commercial use.

---

## 🤝 Contributing

Contributions welcome! To contribute:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## ⭐ Show Your Support

If this project helped you, please give it a star! ⭐

---

## 👨‍💻 Author

Created with ❤️ for the NLP and biomedical research community.

For questions or issues, [open a GitHub issue](https://github.com/TheBiggyBang1/nlp_project/issues).

---

<div align="center">

**Made with ❤️ • [Report Bug](https://github.com/TheBiggyBang1/nlp_project/issues) • [Request Feature](https://github.com/TheBiggyBang1/nlp_project/issues)**

</div>