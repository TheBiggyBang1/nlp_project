# 🏥 Medical NLP Pipeline: PubMed Literature Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Ready-0db7ed?style=flat-square&logo=docker)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

**A comprehensive end-to-end NLP pipeline for biomedical literature analysis from PubMed**

[🚀 Quick Start](#-quick-start) • [📖 Usage Guide](#-usage-guide) • [📁 Structure](#-project-structure) • [🤝 Contributing](#contributing)

</div>

---

A complete Python data science project for analyzing biomedical literature from PubMed using natural language processing and machine learning. This pipeline integrates multiple NLP techniques to extract knowledge and detect healthcare trends from medical abstracts.

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Dependencies](#-dependencies)
- [Troubleshooting](#-troubleshooting)
- [Future Enhancements](#-future-enhancements)

---

## 📋 Project Overview

This project provides an **end-to-end solution** for:
- **Fetching** biomedical literature from PubMed using the NCBI Entrez API
- **Preprocessing** medical abstracts with NLTK and spaCy
- **Analyzing** document characteristics and trends (EDA)
- **Extracting** medical named entities using spaCy NER
- **Modeling** latent topics with Gensim LDA
- **Computing** semantic similarity with Word2Vec
- **Classifying** abstracts into medical specialties using scikit-learn

All components are wrapped in an **interactive Streamlit web application** and containerized with **Docker**.

---

## ✨ Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Data Collection** | Fetch abstracts from PubMed via NCBI Entrez API | ✅ |
| **Exploratory Analysis** | Statistical analysis, trends, word clouds | ✅ |
| **Text Preprocessing** | NLTK + spaCy tokenization, lemmatization, POS filtering | ✅ |
| **Named Entity Recognition** | Extract medical entities with spaCy | ✅ |
| **Topic Modeling** | Gensim LDA with coherence scoring | ✅ |
| **Semantic Similarity** | Word2Vec embeddings for medical terms | ✅ |
| **Classification** | ML-based medical specialty detection | ✅ |
| **Web Interface** | Interactive 7-page Streamlit application | ✅ |
| **Containerization** | Docker & Docker Compose support | ✅ |

### 🎯 Core Capabilities

<details open>
<summary><b>📊 Data Collection</b></summary>

- ✅ Fetch abstracts from PubMed via Biopython Entrez API
- ✅ Rate limiting and retry logic for reliable data collection
- ✅ Extract: PMID, Title, Abstract, Publication Date, Journal, Keywords
- ✅ Save to CSV format

</details>

<details open>
<summary><b>🔍 Exploratory Data Analysis</b></summary>

- ✅ Dataset statistics and missing value analysis
- ✅ Abstract length distribution
- ✅ Publication trends by year
- ✅ Most frequent terms
- ✅ Word clouds
- ✅ Interactive visualizations

</details>

<details open>
<summary><b>🧹 Text Preprocessing</b></summary>

- ✅ HTML/URL/citation removal
- ✅ Tokenization with NLTK
- ✅ Custom academic stopword removal
- ✅ Lemmatization with spaCy
- ✅ POS-tag filtering (NOUN, PROPN, ADJ)
- ✅ Batch processing for efficiency

</details>

<details open>
<summary><b>🏷️ Named Entity Recognition</b></summary>

- ✅ Extract medical entities using spaCy
- ✅ Fallback to standard model if biomedical model unavailable
- ✅ Entity frequency analysis
- ✅ Entity label distribution
- ✅ Visualizations and statistics

</details>

<details open>
<summary><b>📚 Topic Modeling</b></summary>

- ✅ Gensim LDA implementation
- ✅ Configurable number of topics (2-10)
- ✅ Coherence score calculation
- ✅ Topic-document distributions
- ✅ Top words per topic visualization

</details>

<details open>
<summary><b>🔗 Semantic Similarity</b></summary>

- ✅ Word2Vec model training
- ✅ Find similar medical terms
- ✅ Word analogy support
- ✅ Vocabulary browser

</details>

<details open>
<summary><b>🤖 Medical Classification</b></summary>

- ✅ Auto-label abstracts (Oncology, Cardiology, Infectious Disease, Neurology, Other)
- ✅ Random Forest and SVM classifiers
- ✅ TF-IDF feature extraction
- ✅ Confusion matrix, F1-score, classification reports
- ✅ Per-sample predictions with confidence scores

</details>

<details open>
<summary><b>🌐 Web Interface</b></summary>

- ✅ 7-page Streamlit application
- ✅ Real-time data processing
- ✅ Caching for performance
- ✅ Download results as CSV
- ✅ Responsive design

</details>

---

## 🚀 Quick Start

### ⚙️ Option 1: Docker (Recommended) 🐳

**Requirements:**
- Docker and Docker Compose installed
- ~5-10 minutes for first-time build

**Setup:**

```bash
# Clone or navigate to project directory
cd pubmed-medical-nlp

# Build and run with Docker Compose
docker compose up --build

# The app will be available at:
# http://localhost:8501
```

Once the build completes (~30-60 seconds), open your browser and navigate to `http://localhost:8501`.

**To stop the app:**
```bash
docker compose down
```

---

### 🐍 Option 2: Local Installation (Python 3.11+)

**Requirements:**
- Python 3.11 or higher
- pip package manager
- ~15 minutes setup time

**Setup:**

```bash
# 1. Clone the repository and navigate to it
cd pubmed-medical-nlp

# 2. Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download required NLP models
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords

# 5. Set Entrez email (required by PubMed API)
# On Windows (PowerShell):
$env:ENTREZ_EMAIL="your-email@example.com"

# On macOS/Linux:
export ENTREZ_EMAIL="your-email@example.com"

# 6. Run Streamlit app
streamlit run app.py
```

The app will automatically open at `http://localhost:8501`.

---

## 📖 Usage Guide

### 📄 Page 1: Dataset & PubMed Search

<div align="center"><strong>Fetch and explore biomedical literature</strong></div>

1. **Enter a search query** (e.g., "cancer immunotherapy", "COVID-19 vaccine")
2. **Choose number of abstracts** (10-10,000)
3. **Click "Fetch PubMed Data"**
4. **View sample data** and summary statistics
5. **Download CSV** if needed

**💡 Default:** "cancer immunotherapy" with 100 abstracts

---

### 📊 Page 2: Exploratory Data Analysis

Understand your dataset through statistical analysis and visualizations:

- 📈 Dataset statistics and missing value analysis
- 📏 Abstract length distribution
- 📅 Publication trends by year
- 🔤 Most frequent terms
- ☁️ Word clouds
- 📉 Visual missing value patterns

---

### 🧹 Page 3: Text Preprocessing

Transform raw text into clean, analyzable tokens:

1. **Click "Process All Abstracts"**
2. **View before/after comparison** (original vs. processed)
3. **Check preprocessing statistics** (tokens removed, reduced)
4. **Download processed dataset** as CSV

**Processing Pipeline:**
- HTML/URL/citation removal
- Tokenization & word segmentation
- Stopword filtering (custom academic list)
- Lemmatization
- POS tagging & filtering (NOUN, PROPN, ADJ only)

---

### 🏷️ Page 4: Named Entity Recognition

Extract medical entities using state-of-the-art NLP:

1. **Click "Extract Named Entities"**
2. **Explore entity distribution** by label type
3. **View top entities** in your dataset
4. **See sample entities** from random abstracts
5. **Access statistics** and frequency counts

**Entity Types:** Medical terms, locations, organizations, persons, events, and more

---

### 📚 Page 5: Topic Modeling

Discover latent themes in medical literature with LDA:

1. **Choose number of topics** (2-10 recommended)
2. **Click "Train LDA Model"**
3. **View top words** for each topic
4. **Check topic distribution** across documents
5. **Query topics** for specific documents

**Performance Metric:** Coherence Score (higher = better topics, range: 0.0-1.0)

**Example Topics:** Treatment mechanisms, Clinical trials, Disease types, Research methods

---

### 🔗 Page 6: Word2Vec Semantic Similarity

Explore semantic relationships between medical terms:

1. **Click "Train Word2Vec Model"**
2. **Enter a medical term** (e.g., "cancer", "vaccine", "diabetes")
3. **Set number of similar words** (5-20)
4. **View similar terms** with similarity scores
5. **Browse complete vocabulary**

**Example Queries:**
- `"cancer"` → oncology, tumor, carcinoma, neoplasm, ...
- `"vaccine"` → immunization, vaccination, antibody, immunity, ...
- `"heart"` → cardiac, cardiovascular, hypertension, ...

---

### 🤖 Page 7: ML Classification

Classify abstracts into medical specialties:

1. **Choose classifier** (Random Forest or SVM)
2. **Click "Train Classifier"**
3. **Review performance metrics** (accuracy, precision, recall)
4. **View confusion matrix**
5. **Make predictions** on sample abstracts

**Medical Specialties:**
- 🔴 **Oncology** - Cancer, tumors, chemotherapy
- 💙 **Cardiology** - Heart, cardiac, hypertension
- 🦠 **Infectious Disease** - Infection, virus, vaccine, COVID
- 🧠 **Neurology** - Brain, neurological, Alzheimer's, stroke
- 📋 **Other** - All remaining topics

---

### 🏷️ Page 4: Named Entity Recognition

## 📁 Project Structure

```
pubmed-medical-nlp/
│
├── 📄 app.py                          # Main Streamlit app (7 pages)
│
├── 📁 src/                            # Core NLP modules
│   ├── __init__.py
│   ├── pubmed_api.py                  # 🌐 PubMed fetching (Biopython Entrez)
│   ├── preprocessing.py               # 🧹 Text preprocessing (NLTK + spaCy)
│   ├── eda.py                         # 📊 Exploratory data analysis
│   ├── ner.py                         # 🏷️ Named entity recognition
│   ├── topic_modeling.py              # 📚 LDA topic modeling (Gensim)
│   ├── word2vec_model.py              # 🔗 Word2Vec semantics (Gensim)
│   └── classifier.py                  # 🤖 Medical classification (scikit-learn)
│
├── 📁 data/                           # Data storage
│   ├── raw/                           # Raw fetched abstracts
│   │   └── pubmed_data.csv
│   └── processed/                     # Cleaned abstracts
│       └── pubmed_processed.csv
│
├── 📁 outputs/                        # Generated results
│   ├── figures/                       # Visualizations & charts
│   │   ├── abstract_length_dist.png
│   │   ├── publication_trend.png
│   │   └── ...
│   └── models/                        # Trained models
│       ├── lda_model.model
│       ├── word2vec_model.model
│       ├── classifier.pkl
│       └── tfidf.pkl
│
├── 📋 requirements.txt                # Python dependencies
├── 🐳 Dockerfile                      # Container image
├── 🐳 docker-compose.yml              # Docker Compose config
├── 📖 README.md                       # This file
└── 🚫 .gitignore                      # Git ignore rules
```

---

## 📦 Dependencies

<details open>
<summary><b>🔍 Core Libraries</b></summary>

| Package | Purpose |
|---------|---------|
| `streamlit` | Interactive web framework |
| `pandas` | Data manipulation & analysis |
| `numpy` | Numerical computing |

</details>

<details open>
<summary><b>📖 NLP & Text Processing</b></summary>

| Package | Purpose |
|---------|---------|
| `nltk` | Natural Language Toolkit |
| `spacy` | Industrial-strength NLP |
| `gensim` | Topic modeling (LDA) & Word2Vec |
| `biopython` | PubMed API access |

</details>

<details open>
<summary><b>🤖 Machine Learning</b></summary>

| Package | Purpose |
|---------|---------|
| `scikit-learn` | ML algorithms & evaluation |

</details>

<details open>
<summary><b>🎨 Visualization</b></summary>

| Package | Purpose |
|---------|---------|
| `matplotlib` | Static plots & charts |
| `seaborn` | Statistical visualization |
| `wordcloud` | Word cloud generation |
| `plotly` | Interactive plots |

</details>

<details open>
<summary><b>⚙️ Utilities</b></summary>

| Package | Purpose |
|---------|---------|
| `python-dotenv` | Environment variables |
| `tqdm` | Progress bars |

</details>

See `requirements.txt` for the complete list with versions.

---

## 🔧 Configuration

### Environment Variables

**Required:**
```bash
ENTREZ_EMAIL=your-email@example.com  # Required by NCBI for API access
```

**Optional:**
```bash
STREAMLIT_SERVER_PORT=8501         # Custom port (default: 8501)
STREAMLIT_SERVER_ADDRESS=0.0.0.0   # Server address
```

### Setting Environment Variables

**For Docker:**
- Edit `docker-compose.yml` and set the environment variable under `services` → `pubmed-nlp` → `environment`

**For Local (Windows PowerShell):**
```powershell
$env:ENTREZ_EMAIL="your-email@example.com"
streamlit run app.py
```

**For Local (macOS/Linux):**
```bash
export ENTREZ_EMAIL="your-email@example.com"
streamlit run app.py
```

**Using .env file:**
```bash
# Create .env file in project root
echo ENTREZ_EMAIL="your-email@example.com" > .env

# Then run (python-dotenv will load it automatically)
streamlit run app.py
```

### PubMed API Rate Limiting & Best Practices

The PubMed API enforces strict rate limits:

| Limit | Requests/Second |
|-------|-----------------|
| Without API key | 3 |
| With API key | 10 |

**Our Implementation:**
- ✅ 0.35 second delay between requests
- ✅ Exponential backoff on errors
- ✅ Batch fetching to minimize requests
- ✅ Robust error handling

**For Large-Scale Projects (>100K abstracts):** Consider getting an [NCBI API key](https://www.ncbi.nlm.nih.gov/account/register/) for higher limits.

---

## 📊 Sample Results

### Example: "cancer immunotherapy" Query (100 abstracts)

| Metric | Value |
|--------|-------|
| **Abstracts Retrieved** | 100+ |
| **Avg Abstract Length** | 1,500-2,000 characters |
| **Publication Years** | 2010-2025 |
| **Entity Types Found** | PERSON, ORG, GPE, EVENT, +more |
| **Topics Discovered** | Immunotherapy mechanisms, Cancer types, Treatment outcomes, Clinical trials |
| **Most Similar to "cancer"** | oncology, tumor, carcinoma, malignancy, neoplasm |

---

## 🐛 Troubleshooting

### 🐳 Docker Issues

<details>
<summary><b>❌ "Error response from daemon: Bind for 0.0.0.0:8501 failed"</b></summary>

**Cause:** Port 8501 is already in use

**Solution:**
```yaml
# In docker-compose.yml, change:
ports:
  - "8502:8501"  # Use 8502 instead

# Then rebuild:
docker compose up --build
```

</details>

<details>
<summary><b>❌ "No space left on device"</b></summary>

**Cause:** Docker images/containers consuming disk space

**Solution:**
```bash
# Clean up Docker system
docker system prune -a --volumes

# Then rebuild
docker compose up --build
```

</details>

<details>
<summary><b>❌ "ModuleNotFoundError: No module named spacy"</b></summary>

**Cause:** Docker build incomplete or models not downloaded

**Solution:**
```bash
# Force rebuild with fresh dependencies
docker compose up --build --force-recreate
```

</details>

### 💻 Local Installation Issues

<details>
<summary><b>❌ "No module named spacy"</b></summary>

**Cause:** spaCy package not installed or incomplete installation

**Solution:**
```bash
# Reinstall all dependencies
pip install -r requirements.txt

# Download spaCy models explicitly
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords
```

</details>

<details>
<summary><b>❌ "ENTREZ_EMAIL not set"</b></summary>

**Cause:** Required environment variable not configured

**Solution:**
```bash
# Set environment variable (choose one):

# Windows PowerShell:
$env:ENTREZ_EMAIL="your-email@example.com"

# macOS/Linux:
export ENTREZ_EMAIL="your-email@example.com"

# Then run:
streamlit run app.py
```

</details>

<details>
<summary><b>❌ "ModuleNotFoundError: No module named src"</b></summary>

**Cause:** Running from wrong directory

**Solution:**
```bash
# Make sure you're in the project directory
cd pubmed-medical-nlp

# Then run
streamlit run app.py
```

</details>

### 🌐 PubMed API Issues

<details>
<summary><b>❌ "Error fetching data: 429 Too Many Requests"</b></summary>

**Cause:** Exceeded API rate limits

**Solution:**
1. Wait 1-2 minutes before retrying
2. Try fetching fewer abstracts
3. Request an NCBI API key for higher limits

</details>

<details>
<summary><b>❌ "Error fetching data: Connection timeout"</b></summary>

**Cause:** Network connectivity issue

**Solution:**
1. Check your internet connection
2. Verify PubMed is accessible: https://pubmed.ncbi.nlm.nih.gov/
3. Try again after a few seconds

</details>

<details>
<summary><b>❌ "No results found for query"</b></summary>

**Cause:** Query is too specific or uses invalid syntax

**Solution:**
1. Try a simpler, broader query
2. Use actual medical terms
3. Examples: `"cancer treatment"`, `"COVID-19 vaccine"`, `"diabetes 2020-2025"`

</details>

### 💾 Memory & Performance Issues

<details>
<summary><b>❌ "MemoryError" with large datasets</b></summary>

**Cause:** Processing too many abstracts for available RAM

**Solution:**
1. **Reduce abstract count** in the slider (try 1,000-5,000)
2. **Use machine with more RAM** (16GB+ recommended for 100K+ abstracts)
3. **Process in batches** (modify app.py for advanced processing)

</details>

<details>
<summary><b>❌ Word2Vec/LDA models are slow</b></summary>

**Cause:** These models are computationally intensive

**Solution:**
- Training on 10K+ abstracts may take 5-10 minutes
- Use fewer topics (try 3-5 instead of 10) for faster training
- Be patient - first-time training requires more computation

</details>

---

## ✅ Validation & Testing

### 🧪 Quick Test (2-3 minutes)

1. Run with default query: "cancer immunotherapy" + 100 abstracts
2. Complete entire pipeline: Dataset → Classification
3. Verify all visualizations render correctly

### 🔬 Full Test (10-15 minutes)

1. Fetch 1,000 abstracts
2. Run all analysis modules
3. Train all models (LDA, Word2Vec, Classifier)
4. Verify predictions make sense (e.g., "cancer" → Oncology)

### ✓ Validation Checklist

- ✅ **Data Collection** — CSV has all columns: PMID, Title, Abstract, PublicationDate, Journal, Keywords
- ✅ **Preprocessing** — Token reduction 30-50%, lemmatization working correctly
- ✅ **NER** — Entities extracted, distributed across multiple label types
- ✅ **Topic Modeling** — Coherence >0.4, topics are readable and coherent
- ✅ **Word2Vec** — Query "cancer" returns medical terms (similarity 0.5-0.9)
- ✅ **Classification** — Accuracy >0.5, confusion matrix shows reasonable predictions
- ✅ **Streamlit App** — All 7 pages load without errors

---

## 📈 Advanced Usage

### 🎯 Custom Medical Specialties

Add more classification categories by editing [src/classifier.py](src/classifier.py):

```python
SPECIALTY_KEYWORDS = {
    'Dermatology': {'skin', 'dermatitis', 'eczema', 'psoriasis', ...},
    'Orthopedics': {'bone', 'joint', 'fracture', 'arthritis', ...},
    'Your Specialty': {'keyword1', 'keyword2', ...},
}
```

### 🔬 Biomedical spaCy Model

For better medical entity recognition, install the biomedical NLP model:

```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz
```

Then in [app.py](app.py), update:
```python
# Change from:
ner_extractor = EntityExtractor(model_name='en_core_web_sm')

# To:
ner_extractor = EntityExtractor(model_name='en_core_sci_md')
```

### 💾 Save & Load Trained Models

Models are auto-saved to `outputs/models/`. To load them later:

```python
from gensim.models import LdaModel, Word2Vec
import pickle

# Load LDA Topic Model
lda = LdaModel.load('outputs/models/lda_model.model')

# Load Word2Vec Embeddings
w2v = Word2Vec.load('outputs/models/word2vec_model.model')

# Load Classifier
with open('outputs/models/classifier.pkl', 'rb') as f:
    clf = pickle.load(f)
```

### 📦 Batch Processing

For processing large numbers of abstracts, modify [app.py](app.py) to implement batch job processing for scalability.

---

## 📚 References & Resources

- 🔗 [PubMed API Documentation](https://www.ncbi.nlm.nih.gov/books/NBK25497/)
- 🔗 [Biopython Entrez Guide](https://biopython.org/wiki/Documentation)
- 🔗 [NLTK Documentation](https://www.nltk.org/)
- 🔗 [spaCy Models & Pipelines](https://spacy.io/models)
- 🔗 [Gensim LDA Tutorial](https://radimrehurek.com/gensim/models/ldamodel.html)
- 🔗 [Word2Vec Overview](https://radimrehurek.com/gensim/models/word2vec.html)
- 🔗 [Streamlit Documentation](https://docs.streamlit.io/)

---

## ✨ Code Quality Highlights

- ✅ **Modular Design** — Each component in separate, focused file
- ✅ **Comprehensive Documentation** — Docstrings and inline comments throughout
- ✅ **Error Handling** — Robust exception handling and logging
- ✅ **Type Hints** — Full type annotations for IDE support
- ✅ **Efficient Processing** — Batch processing with spaCy pipelines
- ✅ **Performance Caching** — Streamlit caching for speed
- ✅ **Path Management** — No hardcoded paths (uses pathlib)

---

## 🎯 Future Enhancements

- [ ] 😊 **Sentiment Analysis** — Analyze emotional tone of abstracts
- [ ] 🎨 **Entity Visualization** — Interactive spaCy displacy visualizations
- [ ] 📑 **PDF Reports** — Export analysis results as formatted PDFs
- [ ] 🔍 **Document Similarity** — LSA/LDA-based similarity search
- [ ] 🔌 **REST API** — FastAPI endpoint for programmatic access
- [ ] 🗄️ **Database Storage** — PostgreSQL backend for persistence
- [ ] ⚙️ **Hyperparameter Tuning** — Advanced ML model optimization UI
- [ ] 📊 **Multi-Model Comparison** — Compare results across different algorithms
- [ ] 🔄 **Real-Time Streaming** — Live update capabilities
- [ ] 🌍 **Multi-Language Support** — Process abstracts in multiple languages

---

## 📝 License

MIT License - Feel free to use this project for research, education, and commercial purposes.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. **Fork** this repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

---

## 👨‍💻 Author

Created with ❤️ for the NLP and biomedical research community.

For questions or issues, please open a GitHub issue or contact the maintainers.

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
#   n l p _ p r o j e c t 
 
 