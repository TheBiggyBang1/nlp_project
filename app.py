"""
Medical NLP Pipeline - Streamlit Frontend
A complete interactive application for analyzing biomedical literature from PubMed.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pubmed_api import PubMedFetcher, load_pubmed_data
from preprocessing import TextPreprocessor, save_processed_data
from eda import DataExplorer
from ner import EntityExtractor
from topic_modeling import TopicModeler
from word2vec_model import Word2VecModel
from deep_learning_classifier import DeepLearningClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit configuration
st.set_page_config(
    page_title="Medical NLP Pipeline",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'explorer' not in st.session_state:
    st.session_state.explorer = None
if 'ner_extractor' not in st.session_state:
    st.session_state.ner_extractor = None
if 'topic_modeler' not in st.session_state:
    st.session_state.topic_modeler = None
if 'word2vec_model' not in st.session_state:
    st.session_state.word2vec_model = None
if 'cnn_classifier' not in st.session_state:
    st.session_state.cnn_classifier = None
if 'classifier_metrics' not in st.session_state:
    st.session_state.classifier_metrics = None
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False


# ============================================================================
# PAGE 1: DATASET & PUBMED SEARCH
# ============================================================================
def page_dataset_search():
    """Page for fetching and loading PubMed data."""
    st.markdown('<h2 class="section-header">📊 Dataset & PubMed Search</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_input(
            "Enter PubMed Search Query",
            value="cancer immunotherapy",
            help="Example queries: 'cancer immunotherapy', 'COVID-19 vaccine', 'diabetes treatment'"
        )
    
    with col2:
        num_abstracts = st.slider(
            "Number of Abstracts",
            min_value=10000,
            max_value=25000,
            value=10000,
            step=1000,
            help="Higher numbers take longer to fetch"
        )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Fetch PubMed Data", type="primary", use_container_width=True):
            with st.spinner("Fetching PubMed abstracts..."):
                try:
                    fetcher = PubMedFetcher()
                    output_path = Path('data/raw/pubmed_data.csv')
                    saved_path = fetcher.fetch_and_save(query, num_abstracts, output_path)
                    
                    if saved_path:
                        st.session_state.df = load_pubmed_data(saved_path)
                        st.success(f"✅ Fetched {len(st.session_state.df)} abstracts")
                    else:
                        st.error(f"❌ No results found for query: {query}")
                except Exception as e:
                    st.error(f"❌ Error fetching data: {str(e)}")
    
    with col2:
        if st.button("📂 Load Existing Data", use_container_width=True):
            try:
                csv_path = Path('data/raw/pubmed_data.csv')
                if csv_path.exists():
                    st.session_state.df = load_pubmed_data(csv_path)
                    st.success(f"✅ Loaded {len(st.session_state.df)} abstracts")
                else:
                    st.warning("No existing data found. Please fetch PubMed data first.")
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
    
    with col3:
        if st.session_state.df is not None:
            csv_data = st.session_state.df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_data,
                file_name=f"pubmed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Display data if loaded
    if st.session_state.df is not None:
        st.markdown('<h3 class="section-header">Dataset Summary</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Articles", len(st.session_state.df))
        with col2:
            st.metric("Unique Journals", st.session_state.df['Journal'].nunique())
        with col3:
            non_na = len(st.session_state.df[st.session_state.df['Abstract'] != 'N/A'])
            st.metric("Articles with Abstracts", non_na)
        with col4:
            st.metric("Date Range", f"{st.session_state.df['PublicationDate'].min()} - {st.session_state.df['PublicationDate'].max()}")
        
        st.markdown("**Sample of Fetched Data:**")
        st.dataframe(
            st.session_state.df[['PMID', 'Title', 'Journal', 'PublicationDate']].head(10),
            use_container_width=True,
            height=400
        )


# ============================================================================
# PAGE 2: EXPLORATORY DATA ANALYSIS
# ============================================================================
def page_eda():
    """Page for exploratory data analysis."""
    st.markdown('<h2 class="section-header">📈 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please fetch or load data first in the Dataset & Search page")
        return
    
    # Create explorer
    if st.session_state.explorer is None:
        st.session_state.explorer = DataExplorer(st.session_state.df)
    
    # Stats section
    st.markdown("### Dataset Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    stats = st.session_state.explorer.get_dataset_stats()
    with col1:
        st.metric("Total Articles", stats['total_articles'])
    with col2:
        st.metric("Unique Journals", stats['unique_journals'])
    with col3:
        st.metric("Articles with Content", stats['abstracts_with_content'])
    with col4:
        st.write("**Date Range**")
        st.write(stats['date_range'])
    with col5:
        st.write("**Missing Values**")
        for col, val in stats['missing_values'].items():
            st.write(f"{col}: {val}")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Abstract Length", "📅 Publication Trend", "📝 Top Terms", "☁️ Word Cloud", "❌ Missing Values"]
    )
    
    with tab1:
        st.markdown("### Abstract Length Distribution")
        fig = st.session_state.explorer.plot_abstract_length_distribution()
        st.pyplot(fig)
        
        length_stats = st.session_state.explorer.analyze_abstract_lengths()
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Min", f"{length_stats['min_length']} chars")
        with col2:
            st.metric("Max", f"{length_stats['max_length']} chars")
        with col3:
            st.metric("Mean", f"{length_stats['mean_length']:.0f} chars")
        with col4:
            st.metric("Median", f"{length_stats['median_length']:.0f} chars")
        with col5:
            st.metric("Std Dev", f"{length_stats['std_length']:.0f}")
    
    with tab2:
        st.markdown("### Publication Trend by Year")
        fig = st.session_state.explorer.plot_publication_trend()
        st.pyplot(fig)
    
    with tab3:
        st.markdown("### Top Frequent Terms")
        top_n = st.slider("Number of top terms to display", 5, 50, 20)
        fig = st.session_state.explorer.plot_top_terms(top_n)
        st.pyplot(fig)
    
    with tab4:
        st.markdown("### Word Cloud")
        try:
            fig = st.session_state.explorer.plot_wordcloud()
            if fig:
                st.pyplot(fig)
            else:
                st.info("Install wordcloud package to see word cloud visualization")
        except Exception as e:
            st.warning(f"Could not generate word cloud: {str(e)}")
    
    with tab5:
        st.markdown("### Missing Values Visualization")
        fig = st.session_state.explorer.plot_missing_values()
        st.pyplot(fig)


# ============================================================================
# PAGE 3: TEXT PREPROCESSING PREVIEW
# ============================================================================
def page_preprocessing():
    """Page for text preprocessing preview."""
    st.markdown('<h2 class="section-header">🔧 Text Preprocessing Preview</h2>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please fetch or load data first")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Process All Abstracts", type="primary", use_container_width=True):
            with st.spinner("Processing abstracts..."):
                try:
                    st.session_state.preprocessor = TextPreprocessor()
                    st.session_state.df_processed = st.session_state.preprocessor.process_dataframe(
                        st.session_state.df.copy(),
                        text_column='Abstract',
                        keep_pos=['NOUN', 'PROPN', 'ADJ'],
                        min_tokens=10
                    )
                    
                    # Save processed data
                    save_processed_data(st.session_state.df_processed, Path('data/processed/pubmed_processed.csv'))
                    
                    st.success("✅ Preprocessing completed!")
                except Exception as e:
                    st.error(f"❌ Error during preprocessing: {str(e)}")
    
    with col2:
        if st.session_state.df_processed is not None:
            csv_data = st.session_state.df_processed.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Processed",
                data=csv_data,
                file_name=f"pubmed_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Show before/after examples
    if st.session_state.df_processed is not None:
        st.markdown("### Before/After Comparison")
        
        num_samples = st.slider("Number of samples to display", 1, 5, 3)
        
        for idx in range(min(num_samples, len(st.session_state.df_processed))):
            with st.expander(f"📄 Sample {idx + 1}: {st.session_state.df_processed.iloc[idx]['Title'][:60]}..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original Abstract:**")
                    original = st.session_state.df_processed.iloc[idx]['Abstract']
                    st.text_area("Original", original, height=150, disabled=True)
                
                with col2:
                    st.write("**Cleaned Text:**")
                    cleaned = st.session_state.df_processed.iloc[idx]['cleaned_text']
                    st.text_area("Cleaned", cleaned, height=150, disabled=True)
                
                col1, col2, col3 = st.columns(3)
                original_count = len(st.session_state.df_processed.iloc[idx]['Abstract'].split())
                cleaned_count = st.session_state.df_processed.iloc[idx]['token_count']
                
                with col1:
                    st.metric("Original Tokens", original_count)
                with col2:
                    st.metric("Cleaned Tokens", cleaned_count)
                with col3:
                    reduction = (1 - cleaned_count / original_count) * 100
                    st.metric("Reduction", f"{reduction:.1f}%")
        
        # Statistics
        st.markdown("### Preprocessing Statistics")
        stats = st.session_state.preprocessor.get_statistics(st.session_state.df_processed)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Documents Processed", stats['total_documents'])
        with col2:
            st.metric("Avg Tokens", f"{stats['avg_tokens']:.1f}")
        with col3:
            st.metric("Min Tokens", stats['min_tokens'])
        with col4:
            st.metric("Max Tokens", stats['max_tokens'])
        with col5:
            st.metric("Unique Tokens", stats['total_unique_tokens'])


# ============================================================================
# PAGE 4: NAMED ENTITY RECOGNITION
# ============================================================================
def page_ner():
    """Page for named entity recognition."""
    st.markdown('<h2 class="section-header">🔍 Named Entity Recognition</h2>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please fetch or load data first")
        return
    
    if st.button("🚀 Extract Named Entities", type="primary", use_container_width=True):
        with st.spinner("Extracting entities from abstracts..."):
            try:
                st.session_state.ner_extractor = EntityExtractor()
                st.session_state.df = st.session_state.ner_extractor.extract_from_dataframe(
                    st.session_state.df,
                    text_column='Abstract'
                )
                st.success("✅ Entity extraction completed!")
            except Exception as e:
                st.error(f"❌ Error during entity extraction: {str(e)}")
    
    if st.session_state.ner_extractor is not None:
        # Tabs for different entity views
        tab1, tab2, tab3 = st.tabs(["📊 Entity Distribution", "🏷️ Top Entities", "📋 Entity Details"])
        
        with tab1:
            st.markdown("### Entity Label Distribution")
            fig = st.session_state.ner_extractor.plot_entity_labels_distribution()
            st.pyplot(fig)
        
        with tab2:
            st.markdown("### Top Named Entities")
            top_n = st.slider("Number of top entities to display", 5, 50, 20)
            fig = st.session_state.ner_extractor.plot_top_entities(top_n)
            st.pyplot(fig)
        
        with tab3:
            st.markdown("### Entity Statistics")
            entity_stats = st.session_state.ner_extractor.get_entity_stats()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Unique Entities", entity_stats['total_unique_entities'])
            with col2:
                st.metric("Total Mentions", entity_stats['total_entity_mentions'])
            with col3:
                st.metric("Entity Labels", entity_stats['num_entity_labels'])
            
            st.write("**Entity Label Distribution:**")
            label_dist = entity_stats['entity_label_distribution']
            for label, count in sorted(label_dist.items(), key=lambda x: x[1], reverse=True):
                st.write(f"- {label}: {count}")
        
        # Sample entities
        st.markdown("### Sample Entities from Random Abstract")
        if st.button("🎲 Show Random Sample"):
            import random
            sample_idx = random.randint(0, len(st.session_state.df) - 1)
            sample_entities = st.session_state.ner_extractor.get_sample_entities(sample_idx)
            
            if sample_entities:
                st.write(f"**PMID: {st.session_state.df.iloc[sample_idx]['PMID']}**")
                st.write(f"**Title:** {st.session_state.df.iloc[sample_idx]['Title']}")
                
                entity_df = pd.DataFrame(sample_entities, columns=['Entity', 'Label'])
                st.dataframe(entity_df, use_container_width=True)
            else:
                st.info("No entities found in this sample")


# ============================================================================
# PAGE 5: TOPIC MODELING
# ============================================================================
def page_topic_modeling():
    """Page for LDA topic modeling with automatic topic detection."""
    st.markdown('<h2 class="section-header">🎯 Topic Modeling (LDA)</h2>', unsafe_allow_html=True)
    
    if st.session_state.df_processed is None:
        st.warning("⚠️ Please process abstracts in the Preprocessing page first")
        return
    
    # Sidebar controls
    topic_mode = st.radio("Topic Selection Mode", 
                         ["Manual (Specify Topics)", "Automatic (Detect Optimal)"],
                         horizontal=False)
    
    if topic_mode == "Manual (Specify Topics)":
        num_topics = st.slider("Number of Topics", min_value=2, max_value=10, value=5, step=1)
    else:
        st.info("🤖 The model will automatically find the optimal number of topics by testing multiple configurations and selecting the one with the highest coherence score.")
        col1, col2 = st.columns(2)
        with col1:
            min_topics = st.slider("Min Topics to Test", min_value=2, max_value=5, value=2, step=1)
        with col2:
            max_topics = st.slider("Max Topics to Test", min_value=6, max_value=15, value=10, step=1)
        num_topics = None
    
    num_words = st.slider("Top Words per Topic", min_value=5, max_value=20, value=10, step=1)
    
    if st.button("🚀 Train LDA Model", type="primary", use_container_width=True):
        with st.spinner(f"Training LDA model..."):
            try:
                st.session_state.topic_modeler = TopicModeler()
                
                # Prepare corpus
                tokenized_texts = st.session_state.df_processed['cleaned_tokens'].tolist()
                st.session_state.topic_modeler.prepare_corpus(tokenized_texts)
                
                # Train model
                if topic_mode == "Manual (Specify Topics)":
                    st.session_state.topic_modeler.train(num_topics=num_topics)
                    st.success(f"✅ Model trained with {num_topics} topics!")
                else:
                    optimal_topics, coherence_scores = st.session_state.topic_modeler.find_optimal_topics(
                        min_topics=min_topics,
                        max_topics=max_topics,
                        passes=10
                    )
                    st.success(f"✅ Model trained! Optimal topics: {optimal_topics}")
                    
                    # Display coherence scores
                    st.markdown("#### Coherence Scores by Topic Count")
                    coherence_df = pd.DataFrame(list(coherence_scores.items()), 
                                              columns=['Number of Topics', 'Coherence Score'])
                    st.dataframe(coherence_df, use_container_width=True)
                    
                    # Plot coherence scores
                    fig = st.session_state.topic_modeler.plot_coherence_scores(coherence_scores)
                    st.pyplot(fig)
                
                st.info(f"Coherence Score: {st.session_state.topic_modeler.coherence_score:.4f}")
            except Exception as e:
                st.error(f"❌ Error during model training: {str(e)}")
    
    if st.session_state.topic_modeler is not None:
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Topic Words", "📈 Topic Distribution", "🔎 Document Topics"])
        
        with tab1:
            st.markdown(f"### Top {num_words} Words per Topic")
            fig = st.session_state.topic_modeler.plot_top_words_per_topic(num_words)
            st.pyplot(fig)
        
        with tab2:
            st.markdown("### Average Topic Distribution")
            fig = st.session_state.topic_modeler.plot_topic_distribution()
            st.pyplot(fig)
        
        with tab3:
            st.markdown("### Topic Distribution for Specific Document")
            doc_id = st.number_input(
                "Document ID (0-indexed)",
                min_value=0,
                max_value=len(st.session_state.df_processed) - 1,
                value=0
            )
            
            if st.button("📄 Show Document Topics"):
                doc_topics = st.session_state.topic_modeler.get_document_topics(doc_id)
                
                if doc_topics:
                    st.write(f"**PMID:** {st.session_state.df_processed.iloc[doc_id]['PMID']}")
                    st.write(f"**Title:** {st.session_state.df_processed.iloc[doc_id]['Title']}")
                    
                    # Create visualization
                    topics_list, probs = zip(*doc_topics)
                    topics_labels = [f"Topic {t}" for t in topics_list]
                    
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(topics_labels, probs, color='steelblue', edgecolor='black', alpha=0.7)
                    ax.set_xlabel('Probability', fontsize=12)
                    ax.set_title(f'Topic Distribution for Document {doc_id}', fontsize=14, fontweight='bold')
                    ax.invert_yaxis()
                    ax.grid(True, alpha=0.3, axis='x')
                    st.pyplot(fig)
                    
                    # Show top topics
                    st.write("**Top Topics:**")
                    for topic_id, prob in doc_topics[:3]:
                        st.write(f"- Topic {topic_id}: {prob:.4f}")


# ============================================================================
# PAGE 6: WORD2VEC SEMANTIC SIMILARITY
# ============================================================================
def page_word2vec():
    """Page for Word2Vec semantic similarity."""
    st.markdown('<h2 class="section-header">🔤 Word2Vec Semantic Similarity</h2>', unsafe_allow_html=True)
    
    if st.session_state.df_processed is None:
        st.warning("⚠️ Please process abstracts in the Preprocessing page first")
        return
    
    if st.button("🚀 Train Word2Vec Model", type="primary", use_container_width=True):
        with st.spinner("Training Word2Vec model..."):
            try:
                st.session_state.word2vec_model = Word2VecModel()
                
                tokenized_texts = st.session_state.df_processed['cleaned_tokens'].tolist()
                st.session_state.word2vec_model.train(tokenized_texts)
                
                st.success("✅ Word2Vec model trained!")
                st.info(f"Vocabulary size: {st.session_state.word2vec_model.vocab_size} words")
            except Exception as e:
                st.error(f"❌ Error during model training: {str(e)}")
    
    if st.session_state.word2vec_model is not None:
        st.markdown("### Find Similar Medical Terms")
        
        col1, col2 = st.columns(2)
        
        with col1:
            query_word = st.text_input(
                "Enter a medical term",
                value="cancer",
                help="Example: cancer, vaccine, heart, infection"
            )
        
        with col2:
            topn = st.slider("Number of similar words", 5, 20, 10)
        
        if st.button("🔍 Find Similar Words"):
            similar = st.session_state.word2vec_model.find_similar_words(query_word, topn)
            
            if similar:
                # Create dataframe for display
                words, scores = zip(*similar)
                results_df = pd.DataFrame({
                    'Similar Word': words,
                    'Similarity Score': [f"{score:.4f}" for score in scores]
                })
                
                st.dataframe(results_df, use_container_width=True)
                
                # Plot
                fig = st.session_state.word2vec_model.plot_similar_words(query_word, topn)
                st.pyplot(fig)
            else:
                st.warning(f"Word '{query_word}' not found in vocabulary. Try another term.")
        
        # Vocabulary browser
        with st.expander("📚 Browse Vocabulary"):
            vocab = st.session_state.word2vec_model.get_vocabulary()
            st.write(f"**Vocabulary size:** {len(vocab)} unique terms")
            
            # Show sample
            st.write("**Sample of vocabulary (first 50 terms):**")
            st.write(vocab[:50])


# ============================================================================
# PAGE 7: BERT DEEP LEARNING CLASSIFICATION
# ============================================================================
def page_classification():
    """Page for BERT-based deep learning classification."""
    st.markdown('<h2 class="section-header">🧠 BERT Deep Learning Classification</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🔬 BERT-based Classifier for Medical Abstracts
    This classifier uses a pre-trained **BERT (Bidirectional Encoder Representations from Transformers)** 
    model fine-tuned for medical specialty classification.
    
    **Model Advantages:**
    - **Bidirectional Context:** Understands context from both directions
    - **Pre-trained Knowledge:** Leverages BERT's understanding from massive text corpora
    - **Fine-tuning:** Adapted specifically for medical specialty classification
    - **State-of-the-art Performance:** Superior to traditional CNN and RNN approaches
    - **Better Generalization:** Handles diverse medical terminology effectively
    
    The BERT model is more powerful and accurate than CNN-based approaches, learning rich semantic patterns from biomedical text.
    """)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please fetch or load data first")
        return
    
    # Configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        epochs = st.slider("Training Epochs", min_value=1, max_value=5, value=3, step=1,
                          help="Number of passes through training data (3 is usually optimal for BERT)")
    with col2:
        batch_size = st.slider("Batch Size", min_value=8, max_value=32, value=16, step=4,
                              help="Smaller batches = more GPU memory needed")
    with col3:
        learning_rate = st.selectbox("Learning Rate", 
                                    options=[1e-5, 2e-5, 3e-5, 5e-5],
                                    index=1,
                                    help="2e-5 is commonly used for BERT fine-tuning")
    
    # Train button
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🚀 Train BERT Classifier", type="primary", use_container_width=True, disabled=st.session_state.training_in_progress):
            st.session_state.training_in_progress = True
            
    if st.session_state.training_in_progress:
        with st.spinner("🔬 Training BERT model... This may take a few minutes on CPU..."):
            try:
                st.session_state.cnn_classifier = DeepLearningClassifier(
                    model_name='distilbert-base-uncased',
                    max_len=512,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    num_epochs=epochs
                )
                
                # Filter out NaN/None abstracts before training
                valid_abstracts = [
                    abstract for abstract in st.session_state.df['Abstract'].tolist()
                    if isinstance(abstract, str) and len(str(abstract).strip()) > 0
                ]
                
                if not valid_abstracts:
                    st.error("❌ No valid abstracts found for training. Please check your data.")
                    st.session_state.training_in_progress = False
                    return
                
                st.session_state.classifier_metrics = st.session_state.cnn_classifier.train(
                    abstracts=valid_abstracts,
                    epochs=epochs,
                    verbose=1
                )
                
                # Save model
                model_path = Path('outputs/models')
                st.session_state.cnn_classifier.save_model(model_path)
                
                st.session_state.training_in_progress = False
                st.success("✅ BERT Classifier trained successfully!")
                st.rerun()
            except Exception as e:
                st.session_state.training_in_progress = False
                st.error(f"❌ Error during classifier training: {str(e)}")
                logger.error(f"Classification error: {str(e)}")
    
    # Display results
    if st.session_state.classifier_metrics is not None:
        metrics = st.session_state.classifier_metrics
        
        st.markdown("---")
        st.markdown("### 📊 Model Performance")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Test Accuracy", f"{metrics['test_accuracy']:.4f}", 
                     help="Percentage of correct predictions on unseen test data")
        with col2:
            st.metric("📈 F1-Score", f"{metrics['test_f1']:.4f}",
                     help="Harmonic mean of precision and recall")
        with col3:
            st.metric("🔍 Classes", "5",
                     help="Oncology, Cardiology, Infectious Disease, Neurology, Other")
        
        # Tabs for detailed analysis
        tab1, tab2, tab3 = st.tabs(["🔲 Confusion Matrix", "📊 Classification Report", "🏥 Sample Predictions"])
        
        with tab1:
            st.markdown("### Confusion Matrix")
            st.markdown("Shows how well the model classified each specialty:")
            
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, ax = plt.subplots(figsize=(10, 8))
            class_names = ['Oncology', 'Cardiology', 'Infectious Disease', 'Neurology', 'Other']
            sns.heatmap(metrics['confusion_matrix'], 
                       annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names,
                       yticklabels=class_names,
                       ax=ax)
            ax.set_title('Confusion Matrix - BERT Classifier')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            st.pyplot(fig)
            plt.close()
        
        with tab2:
            st.markdown("### Classification Report")
            st.markdown("Detailed metrics for each medical specialty:")
            st.text(metrics['classification_report'])
        
        with tab3:
            axes[0].grid(True, alpha=0.3)
            
            # Loss
        with tab3:
            st.markdown("### 🔮 Make Predictions on Sample Abstracts")
            st.markdown("Test the BERT model on abstracts from your dataset:")
            
            # Select abstract
            sample_idx = st.slider(
                "Select abstract index",
                0,
                len(st.session_state.df) - 1,
                0,
                key="prediction_slider"
            )
            
            if st.button("🎯 Predict Medical Specialty", type="secondary"):
                sample = st.session_state.df.iloc[sample_idx]
                sample_abstract = sample['Abstract']
                
                label, confidence = st.session_state.cnn_classifier.predict(sample_abstract)
                
                st.markdown(f"**PMID:** {sample['PMID']}")
                st.markdown(f"**Title:** {sample['Title']}")
                st.markdown(f"**Abstract:** {sample_abstract}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🏥 Predicted Specialty", label)
                with col2:
                    st.metric("📊 Confidence Score", f"{confidence:.4f}", 
                             help="Probability of the prediction (0-1)")
                
                # Confidence visualization
                st.markdown("**Confidence Visualization:**")
                confidence_pct = confidence * 100
                st.progress(confidence, text=f"{confidence_pct:.1f}% confident")
        
        # Specialty reference
        with st.expander("📖 Medical Specialty Keywords Reference"):
            st.markdown("These keywords are used for initial labeling:")
            for specialty, keywords in st.session_state.cnn_classifier.SPECIALTY_KEYWORDS.items():
                st.write(f"**{specialty}:**")
                st.caption(", ".join(sorted(keywords)))


# ============================================================================
# MAIN APP NAVIGATION
# ============================================================================
def main():
    """Main app entry point."""
    st.markdown('<h1 class="main-header">🏥 Medical NLP Pipeline</h1>', unsafe_allow_html=True)
    st.markdown("""
    A comprehensive platform for analyzing biomedical literature from PubMed using natural language processing
    and machine learning. This pipeline includes data fetching, preprocessing, NER, topic modeling, semantic similarity,
    and medical specialty classification.
    """)
    
    # Page navigation
    page = st.sidebar.radio(
        "📑 Select Page",
        [
            "Dataset & Search",
            "Exploratory Analysis",
            "Text Preprocessing",
            "Named Entity Recognition",
            "Topic Modeling",
            "Word2Vec Similarity",
            "ML Classification"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About This Project")
    st.sidebar.markdown("""
    This application provides a complete NLP pipeline for biomedical literature analysis:
    
    - **PubMed API Integration** for fetching abstracts
    - **NLTK + spaCy** for text preprocessing
    - **Named Entity Recognition** for extracting medical entities
    - **Gensim LDA** for topic modeling
    - **Word2Vec** for semantic similarity
    - **scikit-learn** for medical specialty classification
    
    All powered by Streamlit and containerized with Docker.
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Project Structure")
    st.sidebar.markdown("""
    - `src/` - Source modules
    - `data/raw/` - Raw PubMed data
    - `data/processed/` - Processed abstracts
    - `outputs/figures/` - Generated visualizations
    - `outputs/models/` - Trained models
    """)
    
    # Route to selected page
    if page == "Dataset & Search":
        page_dataset_search()
    elif page == "Exploratory Analysis":
        page_eda()
    elif page == "Text Preprocessing":
        page_preprocessing()
    elif page == "Named Entity Recognition":
        page_ner()
    elif page == "Topic Modeling":
        page_topic_modeling()
    elif page == "Word2Vec Similarity":
        page_word2vec()
    elif page == "ML Classification":
        page_classification()


if __name__ == "__main__":
    main()
