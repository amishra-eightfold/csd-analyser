"""Text analysis module for CSD Analyzer."""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from collections import Counter
import re
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

logger = logging.getLogger(__name__)

# Initialize required NLTK resources
def initialize_nltk_resources():
    """Initialize NLTK resources required for text analysis."""
    resources = [
        ('punkt', 'tokenizers/punkt'),
        ('stopwords', 'corpora/stopwords'),
        ('wordnet', 'corpora/wordnet'),
        ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger')
    ]
    
    for resource_name, resource_path in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                logger.info(f"Downloading NLTK resource: {resource_name}")
                nltk.download(resource_name, quiet=True)
            except Exception as e:
                logger.error(f"Failed to download NLTK resource {resource_name}: {str(e)}")
                continue

# Initialize NLTK resources on module import
try:
    initialize_nltk_resources()
except Exception as e:
    logger.error(f"NLTK initialization failed: {str(e)}")

class TextAnalyzer:
    """Analyzes text content from support tickets."""
    
    def __init__(self):
        """Initialize text analyzer with NLTK components."""
        try:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            logger.error(f"Failed to initialize NLTK components: {str(e)}")
            # Initialize with empty set if NLTK resources are not available
            self.stop_words = set()
            self.lemmatizer = None
            
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for analysis.
        
        Args:
            text (str): Raw text to process
            
        Returns:
            str: Preprocessed text
        """
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stop words and lemmatize
            tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token not in self.stop_words
            ]
            
            return ' '.join(tokens)
            
        except Exception as e:
            logger.error(f"Text preprocessing failed: {str(e)}")
            return text
            
    def extract_keywords(self, 
                        texts: List[str], 
                        top_n: int = 10) -> List[Dict[str, float]]:
        """
        Extract keywords from a list of texts using TF-IDF.
        
        Args:
            texts (List[str]): List of text documents
            top_n (int): Number of top keywords to return
            
        Returns:
            List[Dict[str, float]]: List of keywords with their TF-IDF scores
        """
        try:
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Generate TF-IDF matrix
            tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
            
            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Calculate average TF-IDF scores across documents
            avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Get top keywords
            top_indices = np.argsort(avg_scores)[-top_n:]
            
            keywords = [
                {
                    'keyword': feature_names[i],
                    'score': float(avg_scores[i])
                }
                for i in top_indices
            ]
            
            return sorted(keywords, key=lambda x: x['score'], reverse=True)
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {str(e)}")
            return []
            
    def identify_topics(self, 
                       texts: List[str], 
                       num_topics: int = 5,
                       top_n_words: int = 5) -> List[Dict[str, Any]]:
        """
        Identify topics in text using clustering.
        
        Args:
            texts (List[str]): List of text documents
            num_topics (int): Number of topics to identify
            top_n_words (int): Number of top words per topic
            
        Returns:
            List[Dict[str, Any]]: List of topics with their top words
        """
        try:
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Generate TF-IDF matrix
            tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=num_topics, random_state=42)
            kmeans.fit(tfidf_matrix)
            
            # Get cluster centers
            centers = kmeans.cluster_centers_
            
            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()
            
            topics = []
            for i, center in enumerate(centers):
                # Get top words for this topic
                top_indices = np.argsort(center)[-top_n_words:]
                top_words = [
                    {
                        'word': feature_names[idx],
                        'weight': float(center[idx])
                    }
                    for idx in top_indices
                ]
                
                topics.append({
                    'topic_id': i,
                    'top_words': sorted(top_words, key=lambda x: x['weight'], reverse=True),
                    'size': np.sum(kmeans.labels_ == i)
                })
            
            return topics
            
        except Exception as e:
            logger.error(f"Topic identification failed: {str(e)}")
            return []
            
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment in text using a simple lexicon-based approach.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Sentiment scores
        """
        try:
            # Simple lexicon of positive and negative words
            positive_words = {
                'good', 'great', 'excellent', 'amazing', 'awesome',
                'fantastic', 'helpful', 'resolved', 'fixed', 'working',
                'thanks', 'thank', 'appreciated', 'satisfied', 'happy'
            }
            
            negative_words = {
                'bad', 'poor', 'terrible', 'awful', 'horrible',
                'broken', 'issue', 'problem', 'error', 'bug',
                'crash', 'failed', 'not working', 'unhappy', 'disappointed'
            }
            
            # Preprocess text
            processed_text = self.preprocess_text(text)
            words = processed_text.split()
            
            # Count sentiment words
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            total_words = len(words)
            
            # Calculate sentiment scores
            if total_words > 0:
                positive_score = positive_count / total_words
                negative_score = negative_count / total_words
                neutral_score = 1 - (positive_score + negative_score)
            else:
                positive_score = negative_score = neutral_score = 0
                
            return {
                'positive': positive_score,
                'negative': negative_score,
                'neutral': neutral_score,
                'compound': positive_score - negative_score
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return {
                'positive': 0,
                'negative': 0,
                'neutral': 1,
                'compound': 0
            }
            
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text using regex patterns.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, List[str]]: Dictionary of entity types and their values
        """
        try:
            entities = {
                'emails': [],
                'urls': [],
                'versions': [],
                'dates': []
            }
            
            # Extract emails
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            entities['emails'] = re.findall(email_pattern, text)
            
            # Extract URLs
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            entities['urls'] = re.findall(url_pattern, text)
            
            # Extract versions (e.g., v1.2.3, 2.0.0)
            version_pattern = r'\b(?:v\d+\.\d+(?:\.\d+)?|\d+\.\d+(?:\.\d+)?)\b'
            entities['versions'] = re.findall(version_pattern, text)
            
            # Extract dates (simple pattern)
            date_pattern = r'\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b'
            entities['dates'] = re.findall(date_pattern, text)
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            return {
                'emails': [],
                'urls': [],
                'versions': [],
                'dates': []
            }
