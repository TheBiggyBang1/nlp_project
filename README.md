# 🏥 Medical NLP Pipeline: PubMed Literature Analysis

This project is an end-to-end solution for analyzing biomedical literature from PubMed using Natural Language Processing (NLP) and Machine Learning (ML) techniques.

---

## 📋 Table of Contents

- [Overview](#-project-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Dependencies](#-dependencies)
- [Troubleshooting](#-troubleshooting)
- [Future Enhancements](#-future-enhancements)

---

## 📋 Project Overview

The pipeline includes:

1. **Data Collection**: Fetch abstracts from PubMed using the NCBI Entrez API.
2. **Preprocessing**: Tokenization, lemmatization, and POS tagging with NLTK and spaCy.
3. **Entity Recognition**: Extraction of medical entities using spaCy NER.
4. **Topic Modeling**: Using Gensim LDA for discovering latent themes.
5. **Semantic Similarity**: Utilizing Word2Vec for medical term relationships.
6. **Classification**: Classifying abstracts into medical specialties with scikit-learn.

All components are integrated into a **Streamlit** web application and **Dockerized** for easy deployment.

---

## ✨ Features

- **Data Collection**: Fetch abstracts with Biopython Entrez API.
- **Text Preprocessing**: Tokenization, lemmatization, and stopword filtering.
- **Named Entity Recognition (NER)**: Extract medical entities.
- **Topic Modeling**: LDA-based topic discovery with coherence scoring.
- **Word2Vec**: Semantic similarity between medical terms.
- **Classification**: Medical specialty classification using ML models.
- **Web Interface**: Interactive app for exploring data and results.
- **Docker Support**: Dockerfile and Compose for containerization.

---

## 🛠️ Quick Start

### Clone the repository:

```bash
git clone https://github.com/your-repo/medical-nlp-pipeline.git
cd medical-nlp-pipeline