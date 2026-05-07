"""
Named Entity Recognition module for biomedical abstracts.
Extracts medical entities using spaCy.
"""

import logging
from typing import List, Dict, Tuple
import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


class EntityExtractor:
    """
    Extract named entities from biomedical text using spaCy.
    """
    
    def __init__(self, model_name: str = 'en_core_web_sm'):
        """
        Initialize entity extractor.
        
        Args:
            model_name: spaCy model to use
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.warning(f"Model {model_name} not found. Download with: python -m spacy download {model_name}")
            raise
        
        self.entities = []
        self.entity_labels = Counter()
        self.unique_entities = Counter()
    
    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of (entity_text, label) tuples
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        return entities
    
    def extract_from_dataframe(self, df: pd.DataFrame, 
                               text_column: str = 'Abstract') -> pd.DataFrame:
        """
        Extract entities from all abstracts in DataFrame.
        
        Args:
            df: DataFrame with abstracts
            text_column: Name of text column
            
        Returns:
            DataFrame with entities column added
        """
        logger.info(f"Extracting entities from {len(df)} documents...")
        
        all_entities = []
        for text in df[text_column]:
            entities = self.extract_entities(text)
            all_entities.append(entities)
        
        # Store results
        df['entities'] = all_entities
        self.entities = all_entities
        
        # Collect all unique entities for frequency analysis
        for entity_list in all_entities:
            for entity_text, label in entity_list:
                self.unique_entities[entity_text.lower()] += 1
                self.entity_labels[label] += 1
        
        logger.info(f"Found {len(self.unique_entities)} unique entities")
        return df
    
    def get_frequent_entities(self, top_n: int = 20) -> List[Tuple[str, int]]:
        """
        Get most frequent entities.
        
        Args:
            top_n: Number of top entities to return
            
        Returns:
            List of (entity, count) tuples
        """
        return self.unique_entities.most_common(top_n)
    
    def get_entities_by_label(self) -> Dict[str, int]:
        """
        Get entity counts by label.
        
        Returns:
            Dictionary mapping label to count
        """
        return dict(self.entity_labels)
    
    def get_entity_stats(self) -> Dict:
        """
        Get overall entity statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_unique_entities': len(self.unique_entities),
            'total_entity_mentions': sum(self.unique_entities.values()),
            'num_entity_labels': len(self.entity_labels),
            'entity_label_distribution': dict(self.entity_labels)
        }
        return stats
    
    def plot_top_entities(self, top_n: int = 20, output_path: Path = None):
        """
        Plot top entities.
        
        Args:
            top_n: Number of top entities to plot
            output_path: Path to save figure
        """
        entities, counts = zip(*self.get_frequent_entities(top_n))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(entities, counts, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Frequency', fontsize=12)
        ax.set_ylabel('Entities', fontsize=12)
        ax.set_title(f'Top {top_n} Most Frequent Named Entities', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {output_path}")
        
        return fig
    
    def plot_entity_labels_distribution(self, output_path: Path = None):
        """
        Plot distribution of entity labels.
        
        Args:
            output_path: Path to save figure
        """
        labels, counts = zip(*self.entity_labels.most_common())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(labels, counts, color='coral', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Entity Label', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Entity Distribution by Label', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {output_path}")
        
        return fig
    
    def get_sample_entities(self, doc_index: int = 0) -> List[Tuple[str, str]]:
        """
        Get entities from a sample document.
        
        Args:
            doc_index: Index of document
            
        Returns:
            List of (entity_text, label) tuples
        """
        if doc_index >= len(self.entities):
            return []
        
        return self.entities[doc_index]
