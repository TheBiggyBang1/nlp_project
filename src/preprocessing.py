"""
Text preprocessing module for biomedical abstracts.
Implements comprehensive cleaning, tokenization, lemmatization, and POS filtering.
"""

import re
import logging
from typing import List, Tuple
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from pathlib import Path

# Download NLTK data if not available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline for biomedical abstracts.
    """
    
    # Custom academic stopwords
    ACADEMIC_STOPWORDS = {
        'study', 'significant', 'patient', 'method', 'result', 'research',
        'based', 'associated', 'showed', 'found', 'showed', 'suggests',
        'objective', 'background', 'conclusion', 'aim', 'objective',
        'purpose', 'assess', 'analyze', 'examine', 'investigate',
        'investigate', 'review', 'examined', 'conducted', 'propose',
        'demonstrated', 'indicated', 'data', 'analysis', 'evaluated'
    }
    
    def __init__(self, model_name: str = 'en_core_web_sm'):
        """
        Initialize preprocessor with spaCy model.
        
        Args:
            model_name: spaCy model to use
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.warning(f"Model {model_name} not found. Download with: python -m spacy download {model_name}")
            raise
        
        # Load stopwords
        self.stopwords = set(stopwords.words('english'))
        self.stopwords.update(self.ACADEMIC_STOPWORDS)
        
        # Compile regex patterns
        self.html_pattern = re.compile(r'<[^>]+>')
        self.url_pattern = re.compile(r'http\S+|www\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.citation_pattern = re.compile(r'\(\d+\)|\[\d+\]')  # (1), [1]
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing HTML, URLs, emails, citations, and noise.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str) or text.lower() == 'n/a':
            return ''
        
        # Remove HTML tags
        text = self.html_pattern.sub('', text)
        
        # Remove URLs
        text = self.url_pattern.sub('', text)
        
        # Remove email addresses
        text = self.email_pattern.sub('', text)
        
        # Remove citations
        text = self.citation_pattern.sub('', text)
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        return text
    
    def preprocess_text(self, text: str, keep_pos: List[str] = None) -> List[str]:
        """
        Complete preprocessing pipeline: clean, tokenize, remove stopwords, lemmatize.
        
        Args:
            text: Raw text
            keep_pos: POS tags to keep (e.g., ['NOUN', 'PROPN', 'ADJ'])
                     If None, keeps all tokens
            
        Returns:
            List of preprocessed tokens
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Clean text
        text = self.clean_text(text)
        
        if not text:
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize and process with spaCy
        doc = self.nlp(text)
        tokens = []
        
        for token in doc:
            # Skip if empty, is punctuation, or is number
            if not token.text.strip() or token.is_punct or token.is_digit:
                continue
            
            # Skip stopwords
            if token.text.lower() in self.stopwords:
                continue
            
            # Filter by POS if specified
            if keep_pos and token.pos_ not in keep_pos:
                continue
            
            # Add lemmatized token
            tokens.append(token.lemma_)
        
        return tokens
    
    def batch_preprocess(self, texts: List[str], 
                        keep_pos: List[str] = None) -> Tuple[List[List[str]], List[int]]:
        """
        Preprocess multiple texts efficiently using spaCy batching.
        
        Args:
            texts: List of text strings
            keep_pos: POS tags to keep
            
        Returns:
            Tuple of (processed tokens, token counts)
        """
        cleaned_texts = [self.clean_text(t).lower() if isinstance(t, str) else '' for t in texts]
        
        all_tokens = []
        token_counts = []
        
        # Process in batches for efficiency
        for doc in self.nlp.pipe(cleaned_texts, batch_size=50):
            tokens = []
            
            for token in doc:
                if not token.text.strip() or token.is_punct or token.is_digit:
                    continue
                
                if token.text.lower() in self.stopwords:
                    continue
                
                if keep_pos and token.pos_ not in keep_pos:
                    continue
                
                tokens.append(token.lemma_)
            
            all_tokens.append(tokens)
            token_counts.append(len(tokens))
        
        return all_tokens, token_counts
    
    def process_dataframe(self, df: pd.DataFrame, 
                         text_column: str = 'Abstract',
                         keep_pos: List[str] = None,
                         min_tokens: int = 10) -> pd.DataFrame:
        """
        Preprocess abstracts in a DataFrame.
        
        Args:
            df: DataFrame with abstracts
            text_column: Name of column containing text
            keep_pos: POS tags to keep
            min_tokens: Minimum token count to keep a document
            
        Returns:
            DataFrame with cleaned_text and token_count columns added
        """
        logger.info(f"Processing {len(df)} documents...")
        
        # Batch process texts
        texts = df[text_column].fillna('').tolist()
        tokens_list, token_counts = self.batch_preprocess(texts, keep_pos)
        
        # Add to dataframe
        df['cleaned_tokens'] = tokens_list
        df['token_count'] = token_counts
        df['cleaned_text'] = [' '.join(tokens) for tokens in tokens_list]
        
        # Filter by minimum token count
        original_len = len(df)
        df = df[df['token_count'] >= min_tokens].copy()
        
        filtered_count = original_len - len(df)
        logger.info(f"Removed {filtered_count} documents with <{min_tokens} tokens")
        logger.info(f"Kept {len(df)} documents")
        
        return df
    
    def get_statistics(self, df: pd.DataFrame) -> dict:
        """
        Get preprocessing statistics.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_documents': len(df),
            'avg_tokens': df['token_count'].mean(),
            'min_tokens': df['token_count'].min(),
            'max_tokens': df['token_count'].max(),
            'median_tokens': df['token_count'].median(),
            'total_unique_tokens': len(set([t for tokens in df['cleaned_tokens'] for t in tokens]))
        }
        return stats


def save_processed_data(df: pd.DataFrame, output_path: Path):
    """
    Save processed data to CSV.
    
    Args:
        df: Processed DataFrame
        output_path: Path to save CSV
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")
