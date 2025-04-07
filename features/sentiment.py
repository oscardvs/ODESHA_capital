"""
Sentiment Analysis Module

This module handles NLP-based sentiment analysis for financial texts
including earnings reports, news articles, and social media content.
"""

import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for transformers and related libraries
try:
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers not installed. NLP sentiment analysis will be limited.")

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not installed. Text embedding features will be limited.")


class SentimentAnalysis:
    """
    Class to handle NLP-based sentiment analysis.
    """
    
    def __init__(self, config_path: str = "../config/settings.yaml"):
        """
        Initialize the sentiment analysis module with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.sentiment_model = self.config.get('features', {}).get('sentiment', {}).get('model', 'finbert')
        self.sentiment_enabled = self.config.get('features', {}).get('sentiment', {}).get('enabled', False)
        
        # Initialize models if enabled
        self.nlp_pipeline = None
        self.embedding_model = None
        
        if self.sentiment_enabled:
            self._initialize_models()
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dict containing configuration
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Return default configuration
            return {
                'features': {
                    'sentiment': {
                        'enabled': False,
                        'model': 'finbert'
                    }
                }
            }
    
    def _initialize_models(self):
        """Initialize NLP models based on configuration."""
        if not self.sentiment_enabled:
            return
            
        if not HAS_TRANSFORMERS:
            logger.error("transformers package is required for sentiment analysis")
            return
            
        try:
            # Initialize sentiment analysis model
            if self.sentiment_model == 'finbert':
                model_name = "ProsusAI/finbert"
            elif self.sentiment_model == 'finbert-tone':
                model_name = "yiyanghkust/finbert-tone"
            elif self.sentiment_model == 'distilbert':
                model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            else:
                model_name = "ProsusAI/finbert"  # Default to finbert
                
            logger.info(f"Loading sentiment model: {model_name}")
            self.nlp_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name
            )
            
            # Initialize embedding model if sentence-transformers is available
            if HAS_SENTENCE_TRANSFORMERS:
                embedding_model_name = "all-MiniLM-L6-v2"  # Good balance of speed and performance
                logger.info(f"Loading embedding model: {embedding_model_name}")
                self.embedding_model = SentenceTransformer(embedding_model_name)
                
        except Exception as e:
            logger.error(f"Error initializing NLP models: {e}")
            self.nlp_pipeline = None
            self.embedding_model = None
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not self.sentiment_enabled or not self.nlp_pipeline:
            return {'label': 'neutral', 'score': 0.5}
            
        try:
            # Truncate text if too long (most models have a token limit)
            max_length = 512
            if len(text.split()) > max_length:
                text = ' '.join(text.split()[:max_length])
            
            # Run sentiment analysis
            result = self.nlp_pipeline(text)[0]
            
            # Standardize output format
            sentiment_result = {
                'label': result['label'].lower(),
                'score': result['score']
            }
            
            # Map different model outputs to standard format
            if self.sentiment_model == 'finbert':
                # FinBERT labels: positive, negative, neutral
                pass  # Already in standard format
            elif self.sentiment_model == 'finbert-tone':
                # FinBERT-tone labels: Positive, Negative, Neutral
                sentiment_result['label'] = sentiment_result['label'].lower()
            elif self.sentiment_model == 'distilbert':
                # DistilBERT labels: POSITIVE, NEGATIVE
                sentiment_result['label'] = sentiment_result['label'].lower()
                if sentiment_result['label'] == 'positive':
                    sentiment_result['score'] = result['score']
                else:
                    sentiment_result['score'] = 1 - result['score']
            
            # Add numeric sentiment score (-1 to 1 scale)
            if sentiment_result['label'] == 'positive':
                sentiment_result['sentiment_score'] = sentiment_result['score']
            elif sentiment_result['label'] == 'negative':
                sentiment_result['sentiment_score'] = -sentiment_result['score']
            else:
                sentiment_result['sentiment_score'] = 0
            
            return sentiment_result
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'label': 'neutral', 'score': 0.5, 'sentiment_score': 0}
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for a text.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array containing embedding vector
        """
        if not self.sentiment_enabled or not self.embedding_model:
            return np.zeros(384)  # Default embedding size for all-MiniLM-L6-v2
            
        try:
            # Truncate text if too long
            max_length = 512
            if len(text.split()) > max_length:
                text = ' '.join(text.split()[:max_length])
            
            # Generate embedding
            embedding = self.embedding_model.encode(text)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            return np.zeros(384)  # Default embedding size
    
    def analyze_earnings_report(self, report_text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of an earnings report.
        
        Args:
            report_text: Text of the earnings report
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not report_text:
            return {'label': 'neutral', 'score': 0.5, 'sentiment_score': 0}
            
        try:
            # Split report into sections
            sections = self._split_into_sections(report_text)
            
            # Analyze each section
            section_sentiments = []
            
            for section_name, section_text in sections.items():
                if not section_text:
                    continue
                    
                sentiment = self.analyze_sentiment(section_text)
                sentiment['section'] = section_name
                section_sentiments.append(sentiment)
            
            # Calculate overall sentiment
            if section_sentiments:
                # Weight the sections (higher weight for summary, outlook, guidance)
                weighted_scores = []
                
                for sentiment in section_sentiments:
                    weight = 1.0
                    section = sentiment['section'].lower()
                    
                    if 'summary' in section or 'highlight' in section:
                        weight = 2.0
                    elif 'outlook' in section or 'guidance' in section or 'future' in section:
                        weight = 3.0
                    
                    weighted_scores.append(sentiment['sentiment_score'] * weight)
                
                # Calculate weighted average
                overall_score = sum(weighted_scores) / sum([1.0, 2.0, 3.0])
                
                # Determine overall label
                if overall_score > 0.1:
                    overall_label = 'positive'
                elif overall_score < -0.1:
                    overall_label = 'negative'
                else:
                    overall_label = 'neutral'
                
                overall_sentiment = {
                    'label': overall_label,
                    'sentiment_score': overall_score,
                    'section_sentiments': section_sentiments
                }
            else:
                # If no sections were analyzed, analyze the whole text
                overall_sentiment = self.analyze_sentiment(report_text)
                overall_sentiment['section_sentiments'] = []
            
            return overall_sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing earnings report: {e}")
            return {'label': 'neutral', 'score': 0.5, 'sentiment_score': 0}
    
    def _split_into_sections(self, text: str) -> Dict[str, str]:
        """
        Split a document into sections.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary mapping section names to section text
        """
        sections = {}
        
        try:
            # Simple section splitting based on common headers
            lines = text.split('\n')
            current_section = 'Introduction'
            current_text = []
            
            for line in lines:
                # Check if line is a potential header
                if line.strip().isupper() or (len(line.strip()) > 0 and line.strip()[-1] == ':'):
                    # Save previous section
                    if current_text:
                        sections[current_section] = '\n'.join(current_text)
                    
                    # Start new section
                    current_section = line.strip().rstrip(':')
                    current_text = []
                else:
                    current_text.append(line)
            
            # Save last section
            if current_text:
                sections[current_section] = '\n'.join(current_text)
            
            return sections
            
        except Exception as e:
            logger.error(f"Error splitting text into sections: {e}")
            return {'Full Text': text}
    
    def analyze_news_headlines(self, headlines: List[str]) -> Dict[str, Any]:
        """
        Analyze sentiment of a list of news headlines.
        
        Args:
            headlines: List of news headlines
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not headlines:
            return {'overall_sentiment': 0, 'headline_sentiments': []}
            
        try:
            # Analyze each headline
            headline_sentiments = []
            
            for headline in headlines:
                sentiment = self.analyze_sentiment(headline)
                sentiment['headline'] = headline
                headline_sentiments.append(sentiment)
            
            # Calculate overall sentiment
            sentiment_scores = [s['sentiment_score'] for s in headline_sentiments]
            overall_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            
            return {
                'overall_sentiment': overall_sentiment,
                'headline_sentiments': headline_sentiments
            }
            
        except Exception as e:
            logger.error(f"Error analyzing news headlines: {e}")
            return {'overall_sentiment': 0, 'headline_sentiments': []}
    
    def analyze_social_media(self, posts: List[str]) -> Dict[str, Any]:
        """
        Analyze sentiment of social media posts.
        
        Args:
            posts: List of social media posts
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not posts:
            return {'overall_sentiment': 0, 'post_sentiments': []}
            
        try:
            # Analyze each post
            post_sentiments = []
            
            for post in posts:
                sentiment = self.analyze_sentiment(post)
                sentiment['post'] = post
                post_sentiments.append(sentiment)
            
            # Calculate overall sentiment
            sentiment_scores = [s['sentiment_score'] for s in post_sentiments]
            overall_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            
            # Calculate sentiment distribution
            positive_count = sum(1 for s in post_sentiments if s['label'] == 'positive')
            negative_count = sum(1 for s in post_sentiments if s['label'] == 'negative')
            neutral_count = sum(1 for s in post_sentiments if s['label'] == 'neutral')
            
            total_count = len(post_sentiments)
            
            sentiment_distribution = {
                'positive': positive_count / total_count if total_count > 0 else 0,
                'negative': negative_count / total_count if total_count > 0 else 0,
                'neutral': neutral_count / total_count if total_count > 0 else 0
            }
            
            return {
                'overall_sentiment': overall_sentiment,
                'sentiment_distribution': sentiment_distribution,
                'post_sentiments': post_sentiments
            }
            
        except Exception as e:
            logger.error(f"Error analyzing social media posts: {e}")
            return {'overall_sentiment': 0, 'post_sentiments': []}
    
    def add_sentiment_features(self, df: pd.DataFrame, sentiment_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Add sentiment features to DataFrame.
        
        Args:
            df: DataFrame to add features to
            sentiment_data: Dictionary with sentiment analysis results
            
        Returns:
            DataFrame with added sentiment features
        """
        if df.empty or not sentiment_data:
            return df
            
        try:
            # Make a copy to avoid modifying the original
            result_df = df.copy()
            
            # Add overall sentiment score
            if 'overall_sentiment' in sentiment_data:
                result_df['sentiment_score'] = sentiment_data['overall_sentiment']
            elif 'sentiment_score' in sentiment_data:
                result_df['sentiment_score'] = sentiment_data['sentiment_score']
            
            # Add sentiment label
            if 'label' in sentiment_data:
                result_df['sentiment_label'] = sentiment_data['label']
            
            # Add sentiment distribution if available
            if 'sentiment_distribution' in sentiment_data:
                dist = sentiment_data['sentiment_distribution']
                result_df['sentiment_positive_ratio'] = dist.get('positive', 0)
                result_df['sentiment_negative_ratio'] = dist.get('negative', 0)
                result_df['sentiment_neutral_ratio'] = dist.get('neutral', 0)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error adding sentiment features: {e}")
            return df


# Example usage
if __name__ == "__main__":
    # This will run if the script is executed directly
    sentiment = SentimentAnalysis()
    
    # Example: Analyze a sample text
    sample_text = """
    The company reported strong earnings for Q1 2023, exceeding analyst expectations.
    Revenue increased by 15% year-over-year, driven by growth in the cloud services segment.
    However, the company faces challenges in the hardware division due to supply chain constraints.
    Management remains optimistic about the full-year outlook and raised guidance for Q2.
    """
    
    result = sentiment.analyze_sentiment(sample_text)
    print("Sentiment Analysis Result:")
    print(result)
    
    # Example: Add sentiment features to DataFrame
    data = {
        'date': [datetime.now()],
        'close': [100]
    }
    
    df = pd.DataFrame(data)
    
    result_df = sentiment.add_sentiment_features(df, result)
    print("\nDataFrame with Sentiment Features:")
    print(result_df)
