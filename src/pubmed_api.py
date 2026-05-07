"""
PubMed API module for fetching biomedical literature abstracts.
Uses Biopython Entrez interface to query NCBI PubMed database.
"""

import os
import time
import logging
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
from pathlib import Path
from Bio import Entrez

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Entrez email (required by NCBI for responsible use)
Entrez.email = os.getenv('ENTREZ_EMAIL', 'user@example.com')


class PubMedFetcher:
    """
    Fetches PubMed abstracts using the Entrez API.
    Implements rate limiting and error handling for reliable data collection.
    """
    
    def __init__(self, delay: float = 1.0):
        """
        Initialize PubMed fetcher.
        
        Args:
            delay: Delay between requests in seconds (1 req/sec = 1.0 delay, safer)
        """
        self.delay = delay
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting (1 request per second for safer Entrez access)."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request_time = time.time()
    
    def search(self, query: str, max_results: int = 100, retries: int = 5) -> List[str]:
        """
        Search PubMed for articles matching the query.
        
        Args:
            query: PubMed search query string
            max_results: Maximum number of results to retrieve
            retries: Number of retry attempts on failure
            
        Returns:
            List of PubMed IDs (PMIDs)
        """
        for attempt in range(retries):
            try:
                self._rate_limit()
                logger.info(f"Searching PubMed for: {query} (max results: {max_results}) [Attempt {attempt + 1}/{retries}]")
                
                handle = Entrez.esearch(
                    db="pubmed",
                    term=query,
                    retmax=max_results,
                    sort='relevance',
                    timeout=30
                )
                record = Entrez.read(handle)
                handle.close()
                
                pmids = record['IdList']
                logger.info(f"Found {len(pmids)} articles for query: {query}")
                return pmids
                
            except Exception as e:
                logger.warning(f"Search attempt {attempt + 1} failed: {str(e)}")
                if attempt < retries - 1:
                    wait_time = (2 ** attempt) * 5  # Exponential backoff: 5, 10, 20, 40, 80 seconds
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to search after {retries} attempts")
                    raise
    
    def fetch_abstracts(self, pmids: List[str], retries: int = 3) -> List[Dict]:
        """
        Fetch full article details including abstracts for given PMIDs.
        
        Args:
            pmids: List of PubMed IDs
            retries: Number of retry attempts on failure
            
        Returns:
            List of article dictionaries with metadata
        """
        articles = []
        
        # Fetch in batches to improve efficiency
        batch_size = 50  # Reduced from 100 for safer batching
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            
            for attempt in range(retries):
                try:
                    self._rate_limit()
                    logger.info(f"Fetching {len(batch)} articles (batch {i//batch_size + 1}/{(len(pmids)-1)//batch_size + 1}) [Attempt {attempt + 1}/{retries}]")
                    
                    id_list = ','.join(batch)
                    handle = Entrez.efetch(
                        db="pubmed",
                        id=id_list,
                        rettype="xml",
                        retmax=batch_size,
                        timeout=30
                    )
                    records = Entrez.read(handle)
                    handle.close()
                    
                    # Parse XML records
                    for article in records['PubmedArticle']:
                        try:
                            parsed = self._parse_article(article)
                            articles.append(parsed)
                        except Exception as e:
                            logger.warning(f"Failed to parse article: {str(e)}")
                    
                    break
                    
                except Exception as e:
                    logger.warning(f"Fetch attempt {attempt + 1} failed: {str(e)}")
                    if attempt < retries - 1:
                        wait_time = (2 ** attempt) * 5  # Exponential backoff: 5, 10, 20 seconds
                        logger.info(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to fetch batch after {retries} attempts")
        
        logger.info(f"Successfully fetched {len(articles)} articles")
        return articles
    
    def _parse_article(self, article: Dict) -> Dict:
        """
        Parse a PubMed article XML record into a dictionary.
        
        Args:
            article: PubMed article record
            
        Returns:
            Dictionary with article metadata
        """
        try:
            medline_citation = article['MedlineCitation']
            article_data = medline_citation['Article']
            pubmed_data = article.get('PubmedData', {})
            
            # Extract basic information
            pmid = medline_citation['PMID']
            title = article_data.get('ArticleTitle', 'N/A')
            
            # Extract abstract (if available)
            abstract_text = 'N/A'
            if 'Abstract' in article_data:
                abstract_obj = article_data['Abstract']
                if isinstance(abstract_obj, dict) and 'AbstractText' in abstract_obj:
                    abstract_parts = abstract_obj['AbstractText']
                    if isinstance(abstract_parts, list):
                        abstract_text = ' '.join([str(part) for part in abstract_parts])
                    else:
                        abstract_text = str(abstract_parts)
            
            # Extract publication date
            pub_date = 'N/A'
            if 'Journal' in article_data and 'JournalIssue' in article_data['Journal']:
                journal_issue = article_data['Journal']['JournalIssue']
                if 'PubDate' in journal_issue:
                    pub_date_obj = journal_issue['PubDate']
                    if 'Year' in pub_date_obj:
                        pub_date = pub_date_obj['Year']
            
            # Extract journal name
            journal = 'N/A'
            if 'Journal' in article_data and 'Title' in article_data['Journal']:
                journal = article_data['Journal']['Title']
            
            # Extract keywords
            keywords = 'N/A'
            if 'KeywordList' in medline_citation and len(medline_citation['KeywordList']) > 0:
                kw_list = medline_citation['KeywordList'][0]
                if 'Keyword' in kw_list:
                    keywords = '; '.join([str(kw) for kw in kw_list['Keyword']])
            
            return {
                'PMID': str(pmid),
                'Title': title,
                'Abstract': abstract_text,
                'PublicationDate': pub_date,
                'Journal': journal,
                'Keywords': keywords
            }
            
        except Exception as e:
            logger.error(f"Error parsing article: {str(e)}")
            raise
    
    def save_to_csv(self, articles: List[Dict], output_path: Path) -> Path:
        """
        Save fetched articles to CSV file.
        
        Args:
            articles: List of article dictionaries
            output_path: Path to save CSV file
            
        Returns:
            Path to saved CSV file
        """
        try:
            df = pd.DataFrame(articles)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(articles)} articles to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save CSV: {str(e)}")
            raise
    
    def fetch_and_save(self, query: str, max_results: int = 100, 
                       output_path: Optional[Path] = None) -> Path:
        """
        Complete pipeline: search, fetch, and save articles.
        
        Args:
            query: PubMed search query
            max_results: Maximum number of articles to fetch
            output_path: Path to save CSV (default: data/raw/pubmed_data.csv)
            
        Returns:
            Path to saved CSV file
        """
        if output_path is None:
            output_path = Path('data/raw/pubmed_data.csv')
        
        # Search for articles
        pmids = self.search(query, max_results)
        
        if not pmids:
            logger.warning(f"No results found for query: {query}")
            return None
        
        # Fetch article details
        articles = self.fetch_abstracts(pmids)
        
        # Save to CSV
        saved_path = self.save_to_csv(articles, output_path)
        
        return saved_path


def load_pubmed_data(csv_path: Path) -> pd.DataFrame:
    """
    Load PubMed data from CSV file with validation.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Pandas DataFrame with PubMed data
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_cols = ['PMID', 'Title', 'Abstract', 'PublicationDate', 'Journal']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Loaded {len(df)} articles from {csv_path}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load PubMed data: {str(e)}")
        raise
