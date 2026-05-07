"""
Exploratory Data Analysis module for PubMed abstracts.
Generates statistics, visualizations, and insights about the dataset.
"""

import logging
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from pathlib import Path

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set matplotlib style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class DataExplorer:
    """
    Exploratory data analysis for PubMed dataset.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize explorer with dataframe.
        
        Args:
            df: DataFrame with PubMed data
        """
        self.df = df.copy()
    
    def get_dataset_stats(self) -> Dict:
        """
        Get basic dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_articles': len(self.df),
            'total_columns': len(self.df.columns),
            'date_range': f"{self.df['PublicationDate'].min()} - {self.df['PublicationDate'].max()}",
            'missing_values': self.df.isnull().sum().to_dict(),
            'abstracts_with_content': len(self.df[self.df['Abstract'] != 'N/A']),
            'unique_journals': self.df['Journal'].nunique()
        }
        return stats
    
    def analyze_abstract_lengths(self) -> Dict:
        """
        Analyze abstract text lengths.
        
        Returns:
            Dictionary with length statistics
        """
        # Calculate abstract lengths (excluding 'N/A')
        lengths = self.df[self.df['Abstract'] != 'N/A']['Abstract'].str.len()
        
        stats = {
            'min_length': int(lengths.min()),
            'max_length': int(lengths.max()),
            'mean_length': float(lengths.mean()),
            'median_length': float(lengths.median()),
            'std_length': float(lengths.std())
        }
        return stats
    
    def get_publication_trend(self) -> Tuple[List, List]:
        """
        Get publication counts by year.
        
        Returns:
            Tuple of (years, counts)
        """
        # Convert to numeric, coerce errors to NaN
        pub_dates = pd.to_numeric(self.df['PublicationDate'], errors='coerce')
        
        # Filter valid years
        pub_dates = pub_dates[(pub_dates >= 1900) & (pub_dates <= 2100)]
        
        # Group by year
        trend = pub_dates.value_counts().sort_index()
        
        return trend.index.tolist(), trend.values.tolist()
    
    def extract_frequent_terms(self, top_n: int = 50) -> Tuple[List, List]:
        """
        Extract most frequent terms from abstracts (raw, no preprocessing).
        
        Args:
            top_n: Number of top terms to return
            
        Returns:
            Tuple of (terms, frequencies)
        """
        # Tokenize all abstracts
        all_words = []
        for abstract in self.df[self.df['Abstract'] != 'N/A']['Abstract']:
            try:
                words = word_tokenize(abstract.lower())
                # Simple filtering: keep words > 3 chars, no punctuation
                words = [w for w in words if w.isalpha() and len(w) > 3]
                all_words.extend(words)
            except:
                continue
        
        # Count frequencies
        counter = Counter(all_words)
        top_terms = counter.most_common(top_n)
        
        terms, freqs = zip(*top_terms) if top_terms else ([], [])
        return list(terms), list(freqs)
    
    def plot_abstract_length_distribution(self, output_path: Path = None):
        """
        Create histogram of abstract lengths.
        
        Args:
            output_path: Path to save figure
        """
        lengths = self.df[self.df['Abstract'] != 'N/A']['Abstract'].str.len()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(lengths, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Abstract Length (characters)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Abstract Lengths', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {output_path}")
        
        return fig
    
    def plot_publication_trend(self, output_path: Path = None):
        """
        Create line plot of publication trend by year.
        
        Args:
            output_path: Path to save figure
        """
        years, counts = self.get_publication_trend()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(years, counts, marker='o', linestyle='-', linewidth=2, 
                markersize=4, color='steelblue')
        ax.set_xlabel('Publication Year', fontsize=12)
        ax.set_ylabel('Number of Articles', fontsize=12)
        ax.set_title('Publication Trend by Year', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {output_path}")
        
        return fig
    
    def plot_top_terms(self, top_n: int = 20, output_path: Path = None):
        """
        Create bar chart of top frequent terms.
        
        Args:
            top_n: Number of top terms to plot
            output_path: Path to save figure
        """
        terms, freqs = self.extract_frequent_terms(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(terms, freqs, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Frequency', fontsize=12)
        ax.set_ylabel('Terms', fontsize=12)
        ax.set_title(f'Top {top_n} Most Frequent Terms', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {output_path}")
        
        return fig
    
    def plot_wordcloud(self, output_path: Path = None):
        """
        Create word cloud from abstracts.
        
        Args:
            output_path: Path to save figure
        """
        try:
            from wordcloud import WordCloud
        except ImportError:
            logger.warning("wordcloud not installed. Install with: pip install wordcloud")
            return None
        
        # Combine all abstracts
        text = ' '.join(self.df[self.df['Abstract'] != 'N/A']['Abstract'].astype(str))
        
        # Generate word cloud
        wordcloud = WordCloud(width=1200, height=600, 
                             background_color='white',
                             colormap='viridis',
                             max_words=100).generate(text)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title('Word Cloud from Abstracts', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {output_path}")
        
        return fig
    
    def plot_missing_values(self, output_path: Path = None):
        """
        Create visualization of missing values.
        
        Args:
            output_path: Path to save figure
        """
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        fig, ax = plt.subplots(figsize=(10, 6))
        missing_pct[missing_pct > 0].plot(kind='barh', ax=ax, color='coral', edgecolor='black')
        ax.set_xlabel('Percentage of Missing Values (%)', fontsize=12)
        ax.set_title('Missing Values by Column', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {output_path}")
        
        return fig
