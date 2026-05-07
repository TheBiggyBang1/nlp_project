"""
Topic Modeling module using Gensim LDA.
Trains and analyzes LDA topic models on biomedical abstracts.
Includes automatic topic number detection using coherence optimization.
"""

import logging
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


class TopicModeler:
    """
    Train and analyze LDA topic models on biomedical texts.
    """
    
    def __init__(self):
        """Initialize topic modeler."""
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.documents = []
        self.coherence_score = None
    
    def prepare_corpus(self, tokenized_texts: List[List[str]], 
                       min_docs: int = 2, max_docs_ratio: float = 0.7):
        """
        Prepare dictionary and corpus from tokenized texts.
        
        Args:
            tokenized_texts: List of tokenized documents
            min_docs: Minimum document frequency
            max_docs_ratio: Maximum document frequency ratio (0-1)
        """
        logger.info("Creating dictionary and corpus...")
        
        # Create dictionary
        self.dictionary = corpora.Dictionary(tokenized_texts)
        
        # Filter extremes
        max_docs = int(len(tokenized_texts) * max_docs_ratio)
        self.dictionary.filter_extremes(no_below=min_docs, no_above=max_docs)
        
        # Create corpus (bag of words)
        self.corpus = [self.dictionary.doc2bow(text) for text in tokenized_texts]
        self.documents = tokenized_texts
        
        logger.info(f"Dictionary size: {len(self.dictionary)}")
        logger.info(f"Corpus size: {len(self.corpus)}")
    
    def find_optimal_topics(self, min_topics: int = 2, max_topics: int = 10, 
                           passes: int = 10, step: int = 1) -> Tuple[int, Dict[int, float]]:
        """
        Find optimal number of topics using coherence score optimization.
        Tests different topic numbers and returns the one with highest coherence.
        
        Args:
            min_topics: Minimum number of topics to test
            max_topics: Maximum number of topics to test
            passes: Number of passes through corpus during training
            step: Step size for topic range (e.g., step=1 tests 2,3,4,5...)
            
        Returns:
            Tuple of (optimal_num_topics, {num_topics: coherence_score})
        """
        if self.corpus is None or self.dictionary is None:
            raise ValueError("Must call prepare_corpus() first")
        
        logger.info(f"Finding optimal number of topics ({min_topics} to {max_topics})...")
        
        coherence_scores = {}
        models = {}
        
        for num_topics in range(min_topics, max_topics + 1, step):
            logger.info(f"Training model with {num_topics} topics...")
            
            model = LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=num_topics,
                random_state=42,
                passes=passes,
                per_word_topics=True,
                minimum_probability=0.0
            )
            
            # Calculate coherence score
            coherence_model = CoherenceModel(
                model=model,
                texts=self.documents,
                dictionary=self.dictionary,
                coherence='c_v'
            )
            coherence_score = coherence_model.get_coherence()
            coherence_scores[num_topics] = coherence_score
            models[num_topics] = model
            
            logger.info(f"Topics: {num_topics}, Coherence Score: {coherence_score:.4f}")
        
        # Find optimal number of topics
        optimal_topics = max(coherence_scores, key=coherence_scores.get)
        logger.info(f"\nOptimal number of topics: {optimal_topics} (Coherence: {coherence_scores[optimal_topics]:.4f})")
        
        # Set the model to the optimal one
        self.lda_model = models[optimal_topics]
        self.coherence_score = coherence_scores[optimal_topics]
        
        return optimal_topics, coherence_scores
    
    def train(self, num_topics: int = 5, passes: int = 10, 
              workers: int = 4) -> LdaModel:
        """
        Train LDA topic model.
        
        Args:
            num_topics: Number of topics
            passes: Number of passes through corpus
            workers: Number of workers (may not be used in all Gensim versions)
            
        Returns:
            Trained LDA model
        """
        if self.corpus is None or self.dictionary is None:
            raise ValueError("Must call prepare_corpus() first")
        
        logger.info(f"Training LDA model with {num_topics} topics...")
        
        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=passes,
            per_word_topics=True,
            minimum_probability=0.0
        )
        
        # Calculate coherence score
        self.coherence_score = CoherenceModel(
            model=self.lda_model,
            texts=self.documents,
            dictionary=self.dictionary,
            coherence='c_v'
        ).get_coherence()
        
        logger.info(f"Model trained. Coherence Score: {self.coherence_score:.4f}")
        
        return self.lda_model
    
    def get_topics(self, top_words: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get top words for each topic.
        
        Args:
            top_words: Number of top words per topic
            
        Returns:
            Dictionary mapping topic_id to [(word, weight), ...]
        """
        if self.lda_model is None:
            raise ValueError("Must train model first")
        
        topics = {}
        for topic_id in range(self.lda_model.num_topics):
            words = self.lda_model.show_topic(topic_id, topn=top_words)
            topics[topic_id] = words
        
        return topics
    
    def get_document_topics(self, doc_id: int) -> List[Tuple[int, float]]:
        """
        Get topic distribution for a single document.
        
        Args:
            doc_id: Document index
            
        Returns:
            List of (topic_id, probability) tuples
        """
        if self.lda_model is None:
            raise ValueError("Must train model first")
        
        if doc_id >= len(self.corpus):
            return []
        
        topics = self.lda_model.get_document_topics(self.corpus[doc_id])
        return sorted(topics, key=lambda x: x[1], reverse=True)
    
    def get_all_document_topics(self) -> np.ndarray:
        """
        Get topic distribution for all documents.
        
        Returns:
            NumPy array of shape (num_docs, num_topics)
        """
        if self.lda_model is None:
            raise ValueError("Must train model first")
        
        num_docs = len(self.corpus)
        num_topics = self.lda_model.num_topics
        doc_topics = np.zeros((num_docs, num_topics))
        
        for doc_id in range(num_docs):
            topics = self.get_document_topics(doc_id)
            for topic_id, prob in topics:
                doc_topics[doc_id, topic_id] = prob
        
        return doc_topics
    
    def save_model(self, model_path: Path):
        """
        Save trained model.
        
        Args:
            model_path: Path to save model
        """
        if self.lda_model is None:
            raise ValueError("No model to save")
        
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.lda_model.save(str(model_path))
        logger.info(f"Saved model to {model_path}")
    
    def load_model(self, model_path: Path):
        """
        Load trained model.
        
        Args:
            model_path: Path to model file
        """
        self.lda_model = LdaModel.load(str(model_path))
        logger.info(f"Loaded model from {model_path}")
    
    def plot_top_words_per_topic(self, top_words: int = 10, 
                                 output_path: Path = None):
        """
        Visualize top words for each topic.
        
        Args:
            top_words: Number of top words per topic
            output_path: Path to save figure
        """
        topics = self.get_topics(top_words)
        num_topics = len(topics)
        
        fig, axes = plt.subplots(
            (num_topics + 1) // 2, 2, 
            figsize=(14, 4 * ((num_topics + 1) // 2))
        )
        axes = axes.flatten()
        
        for topic_id, words in topics.items():
            words_list, weights = zip(*words)
            ax = axes[topic_id]
            ax.barh(words_list, weights, color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Weight', fontsize=10)
            ax.set_title(f'Topic {topic_id}', fontsize=12, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
        
        # Hide unused subplots
        for idx in range(len(topics), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {output_path}")
        
        return fig
    
    def plot_topic_distribution(self, output_path: Path = None):
        """
        Plot average topic distribution across corpus.
        
        Args:
            output_path: Path to save figure
        """
        doc_topics = self.get_all_document_topics()
        avg_distribution = doc_topics.mean(axis=0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        topics = [f'Topic {i}' for i in range(len(avg_distribution))]
        ax.bar(topics, avg_distribution, color='coral', edgecolor='black', alpha=0.7)
        ax.set_ylabel('Average Probability', fontsize=12)
        ax.set_title('Average Topic Distribution Across Corpus', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {output_path}")
        
        return fig
    
    def plot_coherence_scores(self, coherence_scores: Dict[int, float], 
                             output_path: Path = None):
        """
        Plot coherence scores across different topic numbers.
        Helps visualize the optimization curve for topic selection.
        
        Args:
            coherence_scores: Dictionary of {num_topics: coherence_score}
            output_path: Path to save figure
        """
        topics = sorted(coherence_scores.keys())
        scores = [coherence_scores[t] for t in topics]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(topics, scores, 'o-', linewidth=2, markersize=8, color='steelblue')
        
        # Highlight the optimal point
        optimal_topic = max(coherence_scores, key=coherence_scores.get)
        optimal_score = coherence_scores[optimal_topic]
        ax.plot(optimal_topic, optimal_score, 'r*', markersize=20, 
               label=f'Optimal: {optimal_topic} topics')
        
        ax.set_xlabel('Number of Topics', fontsize=12)
        ax.set_ylabel('Coherence Score (C_v)', fontsize=12)
        ax.set_title('Coherence Score Optimization', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        ax.set_xticks(topics)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved coherence plot to {output_path}")
        
        return fig
    
    def create_topic_dataframe(self) -> pd.DataFrame:
        """
        Create DataFrame with document-topic distributions.
        
        Returns:
            DataFrame with columns for each topic
        """
        doc_topics = self.get_all_document_topics()
        
        columns = {f'Topic_{i}': doc_topics[:, i] 
                  for i in range(doc_topics.shape[1])}
        
        return pd.DataFrame(columns)
