import re
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings

# Check if we're in simple mode
SIMPLE_MODE = os.environ.get('SIMPLE_MODE', 'false').lower() == 'true'

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Try to import transformers, but handle the case where it's not available
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    # Check if any deep learning framework is available
    try:
        import torch
        DL_FRAMEWORK = "pytorch"
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        try:
            import tensorflow as tf
            if tf.__version__ >= "2.0.0":
                DL_FRAMEWORK = "tensorflow"
                TRANSFORMERS_AVAILABLE = True
        except ImportError:
            try:
                from jax import numpy as jnp
                import flax
                DL_FRAMEWORK = "flax"
                TRANSFORMERS_AVAILABLE = True
            except ImportError:
                warnings.warn("No deep learning framework (PyTorch, TensorFlow >= 2.0, or Flax) found. "
                             "Using rule-based fallbacks for NLP operations.")
except ImportError:
    warnings.warn("Transformers library not found. Using rule-based fallbacks for NLP operations.")

class NLPUtils:
    """
    Utility class for NLP-related functions.
    Provides text preprocessing, sentiment analysis, and other NLP operations.
    """
    
    @staticmethod
    def preprocess_text(text):
        """
        Preprocess text for NLP analysis.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        # Join tokens back into text
        preprocessed_text = ' '.join(tokens)
        
        return preprocessed_text
    
    @staticmethod
    def extract_keywords(text, num_keywords=5):
        """
        Extract keywords from text.
        
        Args:
            text (str): Input text
            num_keywords (int): Number of keywords to extract
            
        Returns:
            list: List of keywords
        """
        if not text or not isinstance(text, str):
            return []
        
        # Preprocess text
        preprocessed_text = NLPUtils.preprocess_text(text)
        
        # Tokenize
        tokens = word_tokenize(preprocessed_text)
        
        # Count word frequencies
        word_freq = {}
        for word in tokens:
            if len(word) > 2:  # Only consider words with more than 2 characters
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Extract top keywords
        keywords = [word for word, freq in sorted_words[:num_keywords]]
        
        return keywords
    
    @staticmethod
    def detect_emotions(text):
        """
        Detect emotions in text.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary of emotions and their scores
        """
        if not text or not isinstance(text, str):
            return {}
        
        # Define emotion keywords
        emotion_keywords = {
            'joy': ['happy', 'joy', 'delighted', 'glad', 'pleased', 'excited', 'cheerful'],
            'sadness': ['sad', 'unhappy', 'depressed', 'down', 'miserable', 'heartbroken', 'gloomy'],
            'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'enraged', 'hostile'],
            'fear': ['afraid', 'scared', 'frightened', 'terrified', 'anxious', 'nervous', 'worried'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'stunned', 'unexpected'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened', 'appalled', 'horrified']
        }
        
        # Preprocess text
        preprocessed_text = NLPUtils.preprocess_text(text)
        
        # Tokenize
        tokens = word_tokenize(preprocessed_text)
        
        # Count emotion keywords
        emotion_scores = {emotion: 0 for emotion in emotion_keywords}
        
        for token in tokens:
            for emotion, keywords in emotion_keywords.items():
                if token in keywords:
                    emotion_scores[emotion] += 1
        
        # Normalize scores
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total_score
        
        return emotion_scores
    
    @staticmethod
    def detect_mental_health_concerns(text):
        """
        Detect potential mental health concerns in text.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary of concerns and their confidence scores
        """
        if not text or not isinstance(text, str):
            return {}
        
        # Define concern keywords
        concern_keywords = {
            'depression': ['depressed', 'depression', 'sad', 'hopeless', 'worthless', 'empty', 'numb'],
            'anxiety': ['anxious', 'anxiety', 'worried', 'nervous', 'panic', 'fear', 'stress'],
            'suicidal': ['suicide', 'kill myself', 'end my life', 'die', 'death', 'no reason to live'],
            'self_harm': ['cut', 'harm', 'hurt myself', 'pain', 'injury', 'blood', 'scars'],
            'insomnia': ['sleep', 'insomnia', 'awake', 'tired', 'exhausted', 'rest', 'fatigue'],
            'substance_abuse': ['alcohol', 'drugs', 'substance', 'addiction', 'dependent', 'withdrawal']
        }
        
        # Preprocess text
        preprocessed_text = text.lower()  # Keep original text for better matching
        
        # Count concern keywords
        concern_scores = {concern: 0 for concern in concern_keywords}
        
        for concern, keywords in concern_keywords.items():
            for keyword in keywords:
                if keyword in preprocessed_text:
                    concern_scores[concern] += 1
        
        # Normalize scores
        max_score = max(concern_scores.values()) if concern_scores.values() else 0
        if max_score > 0:
            for concern in concern_scores:
                concern_scores[concern] /= max_score
        
        return concern_scores
    
    @staticmethod
    def get_sentiment_analyzer():
        """
        Get a sentiment analysis pipeline.
        
        Returns:
            pipeline: Hugging Face sentiment analysis pipeline or None if not available
        """
        if not TRANSFORMERS_AVAILABLE:
            return None
            
        try:
            # Specify the model explicitly to avoid warnings
            model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
            
            # Load model and tokenizer explicitly
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Create pipeline with explicit model and tokenizer
            return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        except Exception as e:
            warnings.warn(f"Error creating sentiment analyzer: {e}. Using rule-based fallback.")
            return None
    
    @staticmethod
    def analyze_sentiment(text, analyzer=None):
        """
        Analyze sentiment of text.
        
        Args:
            text (str): Input text
            analyzer (pipeline, optional): Sentiment analysis pipeline
            
        Returns:
            dict: Sentiment analysis result
        """
        if not text or not isinstance(text, str):
            return {'label': 'NEUTRAL', 'score': 0.5}
        
        # If transformers is not available or we're in simple mode, use rule-based approach
        if SIMPLE_MODE or not TRANSFORMERS_AVAILABLE:
            return NLPUtils._rule_based_sentiment(text)
        
        # Use provided analyzer or create a new one
        if analyzer is None:
            analyzer = NLPUtils.get_sentiment_analyzer()
            
        # If analyzer creation failed, fall back to rule-based approach
        if analyzer is None:
            return NLPUtils._rule_based_sentiment(text)
        
        try:
            result = analyzer(text)
            return result[0]
        except Exception as e:
            warnings.warn(f"Error analyzing sentiment: {e}. Using rule-based fallback.")
            return NLPUtils._rule_based_sentiment(text)
            
    @staticmethod
    def _rule_based_sentiment(text):
        """
        A simple rule-based sentiment analysis as fallback when transformers is not available.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Sentiment analysis result
        """
        # Define positive and negative word lists
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
            'happy', 'joy', 'love', 'like', 'positive', 'awesome', 'nice',
            'beautiful', 'best', 'better', 'success', 'successful', 'well'
        ]
        
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'poor', 'negative', 
            'sad', 'unhappy', 'hate', 'dislike', 'worst', 'worse', 'fail', 
            'failure', 'problem', 'difficult', 'wrong', 'angry', 'upset'
        ]
        
        # Preprocess text
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        
        # Count positive and negative words
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        # Calculate sentiment score
        total_count = positive_count + negative_count
        if total_count == 0:
            return {'label': 'NEUTRAL', 'score': 0.5}
        
        positive_ratio = positive_count / total_count
        
        # Determine label and score
        if positive_ratio > 0.6:
            return {'label': 'POSITIVE', 'score': 0.5 + positive_ratio / 2}
        elif positive_ratio < 0.4:
            return {'label': 'NEGATIVE', 'score': 0.8 - positive_ratio / 2}
        else:
            return {'label': 'NEUTRAL', 'score': 0.5}
    
    @staticmethod
    def get_sentiment_score(text, analyzer=None):
        """
        Get sentiment score of text (0-1, where 0 is negative and 1 is positive).
        
        Args:
            text (str): Input text
            analyzer (pipeline, optional): Sentiment analysis pipeline
            
        Returns:
            float: Sentiment score
        """
        result = NLPUtils.analyze_sentiment(text, analyzer)
        
        # Convert to a score between 0 and 1
        # where 0 is negative and 1 is positive
        if result['label'] == 'POSITIVE':
            return result['score']
        else:
            return 1 - result['score']
