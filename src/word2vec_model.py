"""
Word2Vec semantic similarity module using Gensim.
Trains Word2Vec models and finds semantically similar terms.
"""

import logging
from typing import List, Tuple
import pandas as pd
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Word2VecModel:
    """
    Train and use Word2Vec models on biomedical abstracts.
    """
    
    def __init__(self):
        """Initialize Word2Vec model."""
        self.model = None
        self.vocab_size = 0
    
    def train(self, tokenized_texts: List[List[str]], 
              vector_size: int = 100, 
              window: int = 5,
              min_count: int = 5, 
              epochs: int = 10,
              workers: int = 4) -> Word2Vec:
        """
        Train Word2Vec model.
        
        Args:
            tokenized_texts: List of tokenized documents
            vector_size: Dimension of word vectors
            window: Context window size
            min_count: Minimum word frequency
            epochs: Number of training epochs
            workers: Number of worker threads
            
        Returns:
            Trained Word2Vec model
        """
        logger.info("Training Word2Vec model...")
        
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            epochs=epochs,
            workers=workers,
            sg=0  # CBOW model (better for smaller datasets)
        )
        
        self.vocab_size = len(self.model.wv)
        logger.info(f"Model trained. Vocabulary size: {self.vocab_size}")
        
        return self.model
    
    def find_similar_words(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Find semantically similar words.
        
        Args:
            word: Query word
            topn: Number of similar words to return
            
        Returns:
            List of (word, similarity_score) tuples
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        word_lower = word.lower()
        
        try:
            similar = self.model.wv.most_similar(word_lower, topn=topn)
            return similar
        except KeyError:
            logger.warning(f"Word '{word}' not in vocabulary")
            return []
    
    def get_vector(self, word: str):
        """
        Get embedding vector for a word.
        
        Args:
            word: Word to get vector for
            
        Returns:
            NumPy array with word vector
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        word_lower = word.lower()
        
        try:
            return self.model.wv[word_lower]
        except KeyError:
            logger.warning(f"Word '{word}' not in vocabulary")
            return None
    
    def get_vocabulary(self) -> List[str]:
        """
        Get model vocabulary.
        
        Returns:
            List of words in vocabulary
        """
        if self.model is None:
            return []
        
        return list(self.model.wv.index_to_key)
    
    def word_analogy(self, positive: List[str], 
                     negative: List[str] = None, topn: int = 5) -> List[Tuple[str, float]]:
        """
        Find words using analogy (e.g., "cancer" is to "oncology" as "heart" is to ?).
        
        Args:
            positive: List of words to add to analogy
            negative: List of words to subtract from analogy
            topn: Number of results to return
            
        Returns:
            List of (word, score) tuples
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        if negative is None:
            negative = []
        
        try:
            results = self.model.wv.most_similar(positive=positive, negative=negative, topn=topn)
            return results
        except KeyError as e:
            logger.warning(f"One or more words not in vocabulary: {e}")
            return []
    
    def save_model(self, model_path: Path):
        """
        Save trained model.
        
        Args:
            model_path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(model_path))
        logger.info(f"Saved Word2Vec model to {model_path}")
    
    def load_model(self, model_path: Path):
        """
        Load trained model.
        
        Args:
            model_path: Path to model file
        """
        self.model = Word2Vec.load(str(model_path))
        self.vocab_size = len(self.model.wv)
        logger.info(f"Loaded Word2Vec model from {model_path}")
    
    def plot_similar_words(self, word: str, topn: int = 10, 
                          output_path: Path = None):
        """
        Plot similar words and their similarity scores.
        
        Args:
            word: Query word
            topn: Number of similar words to plot
            output_path: Path to save figure
        """
        similar = self.find_similar_words(word, topn)
        
        if not similar:
            logger.warning(f"No similar words found for '{word}'")
            return None
        
        words, scores = zip(*similar)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(words, scores, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Similarity Score', fontsize=12)
        ax.set_ylabel('Words', fontsize=12)
        ax.set_title(f'Words Most Similar to "{word}"', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim([0, 1])
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {output_path}")
        
        return fig
