import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import random
import os
import warnings
import openai

# Set OpenAI API key

# Check if we're in simple mode
SIMPLE_MODE = os.environ.get('SIMPLE_MODE', 'false').lower() == 'true'

# Try to import transformers, but handle the case where it's not available
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    # Check if any deep learning framework is available
    try:
        import torch
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        try:
            import tensorflow as tf
            if tf.__version__ >= "2.0.0":
                TRANSFORMERS_AVAILABLE = True
        except ImportError:
            try:
                from jax import numpy as jnp
                import flax
                TRANSFORMERS_AVAILABLE = True
            except ImportError:
                warnings.warn("No deep learning framework found. Using rule-based fallbacks for NLP operations.")
except ImportError:
    warnings.warn("Transformers library not found. Using rule-based fallbacks for NLP operations.")

# Import NLPUtils for fallback sentiment analysis
from modules.utils.nlp_utils import NLPUtils

class ChatbotModule:
    """
    Chatbot Module for providing conversational support to users.
    Uses NLP for intent recognition and sentiment analysis.
    """
    
    def __init__(self, db):
        """
        Initialize the Chatbot Module.
        
        Args:
            db: Database instance for storing chat history
        """
        self.db = db
        
        # Define emergency keywords first to avoid the initialization error
        self.emergency_keywords = [
            "suicide", "kill myself", "end my life", "want to die", 
            "harm myself", "hurt myself", "self-harm", "emergency",
            "crisis", "dangerous", "immediate help"
        ]
        
        # Define intents and responses
        self.intents = self._define_intents()
        
        # Initialize sentiment analysis pipeline if available
        if not SIMPLE_MODE and TRANSFORMERS_AVAILABLE:
            @st.cache_resource
            def load_sentiment_analyzer():
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
            
            self.sentiment_analyzer = load_sentiment_analyzer()
        else:
            self.sentiment_analyzer = None
            if not SIMPLE_MODE:
                st.info("Running without deep learning models. Using rule-based sentiment analysis instead.")
    
    def _define_intents(self):
        """Define chatbot intents and responses."""
        return {
            "greeting": {
                "patterns": ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"],
                "responses": [
                    "Hello! How are you feeling today?",
                    "Hi there! How can I support you today?",
                    "Hello! I'm here to listen and help. How are you doing?",
                    "Hi! How are you feeling right now?"
                ]
            },
            "goodbye": {
                "patterns": ["bye", "goodbye", "see you", "talk to you later", "farewell"],
                "responses": [
                    "Goodbye! Take care of yourself.",
                    "Take care! Remember I'm here if you need to talk.",
                    "Farewell! Wishing you well.",
                    "Goodbye! Remember to be kind to yourself."
                ]
            },
            "thanks": {
                "patterns": ["thank you", "thanks", "appreciate it", "thank you so much", "thanks a lot"],
                "responses": [
                    "You're welcome! I'm glad I could help.",
                    "It's my pleasure to support you.",
                    "You're very welcome. Remember, you're not alone.",
                    "Anytime! I'm here to help."
                ]
            },
            "feeling_bad": {
                "patterns": ["feeling sad", "depressed", "unhappy", "miserable", "feeling down", "feeling low", "not feeling well"],
                "responses": [
                    "I'm sorry to hear you're feeling this way. Would you like to talk about what's bothering you?",
                    "It's okay to feel down sometimes. Would it help to share what's on your mind?",
                    "I'm here for you. Can you tell me more about what's making you feel this way?",
                    "Thank you for sharing that with me. Would you like some suggestions that might help you feel better?"
                ]
            },
            "feeling_anxious": {
                "patterns": ["anxious", "nervous", "worried", "stressed", "panic", "fear", "scared"],
                "responses": [
                    "It sounds like you're feeling anxious. Would you like to try a quick breathing exercise?",
                    "Anxiety can be challenging. What specifically is making you feel this way?",
                    "I understand anxiety can be overwhelming. Would it help to talk about what's causing these feelings?",
                    "When you're feeling anxious, it can help to ground yourself. Would you like me to suggest a grounding technique?"
                ]
            },
            "feeling_good": {
                "patterns": ["feeling good", "happy", "great", "excellent", "wonderful", "amazing", "positive"],
                "responses": [
                    "That's wonderful to hear! What's contributing to your positive mood?",
                    "I'm glad you're feeling good! It's important to acknowledge and celebrate these moments.",
                    "That's great! What positive things have been happening in your life?",
                    "I'm happy to hear that! What strategies have been helping you maintain this positive state?"
                ]
            },
            "need_help": {
                "patterns": ["need help", "can you help", "assist me", "support", "guidance", "advice"],
                "responses": [
                    "I'm here to help. What specific support are you looking for?",
                    "I'd be happy to assist you. Could you tell me more about what you need help with?",
                    "I'm here to support you. What's on your mind?",
                    "I'll do my best to help. What are you struggling with right now?"
                ]
            },
            "about_therapy": {
                "patterns": ["therapy", "therapist", "counseling", "counselor", "psychologist", "psychiatrist", "mental health professional"],
                "responses": [
                    "Therapy can be very beneficial for many people. Would you like information about finding a therapist?",
                    "Working with a mental health professional can provide valuable support. Are you considering therapy?",
                    "Therapy offers a safe space to explore your thoughts and feelings with a trained professional. Would you like to know more about different types of therapy?",
                    "Many people find therapy helpful for managing mental health challenges. Would you like to discuss what to expect from therapy?"
                ]
            },
            "about_meditation": {
                "patterns": ["meditation", "mindfulness", "relaxation techniques", "calm", "relax", "breathing exercises"],
                "responses": [
                    "Meditation can be a helpful tool for managing stress and anxiety. Would you like to learn a simple meditation technique?",
                    "Mindfulness practices can help ground you in the present moment. Would you like some mindfulness exercises to try?",
                    "Regular meditation has been shown to reduce stress and improve well-being. Would you like resources to get started with meditation?",
                    "Breathing exercises can help calm your mind and body. Would you like to try a quick breathing exercise now?"
                ]
            },
            "about_sleep": {
                "patterns": ["sleep", "insomnia", "can't sleep", "trouble sleeping", "nightmares", "tired", "fatigue"],
                "responses": [
                    "Sleep is crucial for mental health. Are you having trouble with your sleep patterns?",
                    "Insomnia can be challenging to deal with. Would you like some tips for improving sleep quality?",
                    "Sleep difficulties can affect your mood and energy levels. Have you noticed any patterns in your sleep disturbances?",
                    "Creating a consistent sleep routine can help improve sleep quality. Would you like some suggestions for a bedtime routine?"
                ]
            },
            "about_assessment": {
                "patterns": ["assessment", "test", "questionnaire", "evaluate", "diagnosis", "self-assessment"],
                "responses": [
                    "Our self-assessment tools can help you better understand your mental health. Would you like to take an assessment?",
                    "Self-assessments can provide insights into your mental well-being. Would you like to try one?",
                    "We offer several types of mental health assessments. Would you like to learn more about them?",
                    "Taking a self-assessment can be a good first step in understanding your mental health needs. Would you like to take one now?"
                ]
            },
            "emergency": {
                "patterns": self.emergency_keywords,
                "responses": [
                    "I'm concerned about what you're sharing. If you're in immediate danger, please call emergency services at 911 or a crisis helpline at 1-800-273-8255. Would you like me to provide more crisis resources?",
                    "Your safety is important. If you're having thoughts of harming yourself, please reach out to a crisis helpline at 1-800-273-8255 or text HOME to 741741 to reach the Crisis Text Line. Would you like to talk about what's happening?",
                    "I'm here for you, but for immediate support during a crisis, please contact emergency services or a mental health crisis line. The National Suicide Prevention Lifeline is available 24/7 at 1-800-273-8255. Can I help you find local crisis resources?",
                    "What you're experiencing sounds serious. Please consider calling the National Suicide Prevention Lifeline at 1-800-273-8255 or texting HOME to 741741. Would it help to discuss some coping strategies for this moment?"
                ]
            },
            "fallback": {
                "patterns": [],
                "responses": [
                    "I'm here to listen. Can you tell me more about what's on your mind?",
                    "I want to understand better. Could you share more about what you're experiencing?",
                    "Thank you for sharing that with me. How can I best support you right now?",
                    "I appreciate you opening up. Would it help to explore this topic further?"
                ]
            }
        }
    
    def detect_intent(self, message):
        """
        Detect the intent of a user message.
        
        Args:
            message (str): User message
            
        Returns:
            str: Detected intent
        """
        message = message.lower()
        
        # Check for emergency keywords first
        for keyword in self.emergency_keywords:
            if keyword in message:
                return "emergency"
        
        # Check other intents
        for intent, data in self.intents.items():
            for pattern in data["patterns"]:
                if pattern in message:
                    return intent
        
        # If no intent is detected, return fallback
        return "fallback"
    
    def analyze_sentiment(self, message):
        """
        Analyze the sentiment of a user message.
        
        Args:
            message (str): User message
            
        Returns:
            float: Sentiment score (0-1, where 0 is negative and 1 is positive)
        """
        # If transformers is not available or we're in simple mode, use rule-based approach
        if SIMPLE_MODE or self.sentiment_analyzer is None:
            result = NLPUtils._rule_based_sentiment(message)
            if result['label'] == 'POSITIVE':
                return result['score']
            else:
                return 1 - result['score']
        
        # Use transformer-based sentiment analysis
        try:
            result = self.sentiment_analyzer(message)
            
            # Convert sentiment to a score between 0 and 1
            # where 0 is negative and 1 is positive
            if result[0]['label'] == 'POSITIVE':
                return result[0]['score']
            else:
                return 1 - result[0]['score']
        except Exception as e:
            warnings.warn(f"Error analyzing sentiment with transformer model: {e}. Using rule-based fallback.")
            # Use rule-based sentiment analysis as fallback
            result = NLPUtils._rule_based_sentiment(message)
            if result['label'] == 'POSITIVE':
                return result['score']
            else:
                return 1 - result['score']
    
    def generate_openai_response(self, message):
        """
        Generate a response using OpenAI's API.
        
        Args:
            message (str): User message
            
        Returns:
            str: Generated response
        """
        try:
            # Create a system message that provides context about the chatbot's role
            system_message = """
            You are a mental health support chatbot. Your role is to provide empathetic, supportive responses to users who may be experiencing mental health challenges. 
            You should be compassionate, non-judgmental, and helpful. You are not a replacement for professional mental health care, but you can offer support, 
            resources, and guidance. If a user appears to be in crisis or mentions self-harm or suicide, prioritize their safety and direct them to appropriate 
            emergency resources. Keep your responses concise (2-3 sentences) and focused on supporting the user's mental well-being.
            """
            
            # Create the conversation with the system message and user message
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": message}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            # Extract the response text
            return response.choices[0].message.content.strip()
        except Exception as e:
            warnings.warn(f"Error generating response with OpenAI: {e}. Using fallback response generation.")
            # Use fallback response generation
            return self.generate_fallback_response(message)
    
    def generate_fallback_response(self, message, intent=None):
        """
        Generate a fallback response based on the user message and intent.
        
        Args:
            message (str): User message
            intent (str, optional): Detected intent
            
        Returns:
            str: Generated response
        """
        # Detect intent if not provided
        if intent is None:
            intent = self.detect_intent(message)
        
        # Analyze sentiment
        sentiment_score = self.analyze_sentiment(message)
        
        # Get responses for the intent
        responses = self.intents[intent]["responses"]
        
        # Select a random response
        response = random.choice(responses)
        
        # Add personalized elements based on sentiment
        if sentiment_score < 0.3 and intent not in ["emergency", "feeling_bad", "feeling_anxious"]:
            response = f"I notice you seem to be feeling down. {response}"
        elif sentiment_score > 0.7 and intent not in ["feeling_good"]:
            response = f"You sound positive! {response}"
        
        return response
    
    def generate_response(self, message):
        """
        Generate a response based on the user message.
        
        Args:
            message (str): User message
            
        Returns:
            tuple: (response, sentiment_score)
        """
        # Check for emergency keywords
        for keyword in self.emergency_keywords:
            if keyword in message.lower():
                # Use fallback response for emergency situations
                return self.generate_fallback_response(message, "emergency"), 0.0
        
        # Try to generate a response using OpenAI
        try:
            response = self.generate_openai_response(message)
            sentiment_score = self.analyze_sentiment(message)
            return response, sentiment_score
        except Exception as e:
            warnings.warn(f"Error generating response with OpenAI: {e}. Using fallback response generation.")
            # Use fallback response generation
            intent = self.detect_intent(message)
            sentiment_score = self.analyze_sentiment(message)
            response = self.generate_fallback_response(message, intent)
            return response, sentiment_score
    
    def render(self):
        """Render the chatbot interface."""
        st.markdown("<div class='main-header'>Chat Support</div>", unsafe_allow_html=True)
        
        # Initialize chat history in session state if not exists
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display information about the chatbot
        st.markdown(
            """
            <div class='info-box'>
            This AI chatbot is here to provide support and guidance. While it can offer helpful resources and suggestions,
            it is not a replacement for professional mental health care. If you're experiencing a crisis, please use the
            emergency contact information in the sidebar.
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Display chat messages
        for message in st.session_state.chat_history:
            if message["is_user"]:
                st.markdown(
                    f"""
                    <div class='user-message'>
                    <strong>You:</strong> {message["text"]}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class='bot-message'>
                    <strong>Support Bot:</strong> {message["text"]}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        # User input
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area("Type your message here:", key="user_input")
            submitted = st.form_submit_button("Send")
            
            if submitted and user_input.strip():
                # Add user message to chat history
                st.session_state.chat_history.append({
                    "text": user_input,
                    "is_user": True,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Generate response
                response, sentiment_score = self.generate_response(user_input)
                
                # Add bot response to chat history
                st.session_state.chat_history.append({
                    "text": response,
                    "is_user": False,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Save to database if user is logged in
                if st.session_state.user_name:
                    user_id = self.db.get_or_create_user(st.session_state.user_name)
                    self.db.save_chat_message(
                        user_id,
                        user_input,
                        response,
                        sentiment_score
                    )
                
                # Rerun to update the chat display
                st.rerun()
        
        # Helpful resources section
        st.markdown("<div class='sub-header'>Helpful Resources</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                """
                <div class='info-box'>
                <strong>Self-Assessment:</strong> Take a mental health assessment to better understand your current state.
                </div>
                """, 
                unsafe_allow_html=True
            )
            if st.button("Take Self-Assessment", key="chat_to_assessment"):
                st.session_state.current_page = 'Self Assessment'
                st.rerun()
        
        with col2:
            st.markdown(
                """
                <div class='info-box'>
                <strong>Resource Library:</strong> Explore articles, videos, and tools for mental health support.
                </div>
                """, 
                unsafe_allow_html=True
            )
            if st.button("Browse Resources", key="chat_to_resources"):
                st.session_state.current_page = 'Resources'
                st.rerun()
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
