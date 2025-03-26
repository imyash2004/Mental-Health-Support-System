import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import warnings

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

class SelfAssessmentModule:
    """
    Self-Assessment Module for mental health evaluation.
    Provides questionnaires and NLP-based analysis of responses.
    """
    
    def __init__(self, db):
        """
        Initialize the Self-Assessment Module.
        
        Args:
            db: Database instance for storing assessment results
        """
        self.db = db
        
        # Initialize sentiment analysis pipeline if available
        # Using a cached resource to avoid loading the model every time
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
        
        # Define assessment types and their questions
        self.assessment_types = {
            "general": {
                "title": "General Mental Health Assessment",
                "description": "This assessment helps evaluate your overall mental well-being.",
                "questions": self._get_general_assessment_questions()
            },
            "anxiety": {
                "title": "Anxiety Assessment",
                "description": "This assessment focuses on symptoms related to anxiety.",
                "questions": self._get_anxiety_assessment_questions()
            },
            "depression": {
                "title": "Depression Assessment",
                "description": "This assessment focuses on symptoms related to depression.",
                "questions": self._get_depression_assessment_questions()
            }
        }
    
    def _get_general_assessment_questions(self):
        """Get questions for the general mental health assessment."""
        return [
            {
                "id": "general_1",
                "text": "Over the past 2 weeks, how often have you felt little interest or pleasure in doing things?",
                "options": [
                    {"value": 0, "text": "Not at all"},
                    {"value": 1, "text": "Several days"},
                    {"value": 2, "text": "More than half the days"},
                    {"value": 3, "text": "Nearly every day"}
                ],
                "type": "multiple_choice"
            },
            {
                "id": "general_2",
                "text": "Over the past 2 weeks, how often have you felt down, depressed, or hopeless?",
                "options": [
                    {"value": 0, "text": "Not at all"},
                    {"value": 1, "text": "Several days"},
                    {"value": 2, "text": "More than half the days"},
                    {"value": 3, "text": "Nearly every day"}
                ],
                "type": "multiple_choice"
            },
            {
                "id": "general_3",
                "text": "Over the past 2 weeks, how often have you felt nervous, anxious, or on edge?",
                "options": [
                    {"value": 0, "text": "Not at all"},
                    {"value": 1, "text": "Several days"},
                    {"value": 2, "text": "More than half the days"},
                    {"value": 3, "text": "Nearly every day"}
                ],
                "type": "multiple_choice"
            },
            {
                "id": "general_4",
                "text": "Over the past 2 weeks, how often have you been unable to stop or control worrying?",
                "options": [
                    {"value": 0, "text": "Not at all"},
                    {"value": 1, "text": "Several days"},
                    {"value": 2, "text": "More than half the days"},
                    {"value": 3, "text": "Nearly every day"}
                ],
                "type": "multiple_choice"
            },
            {
                "id": "general_5",
                "text": "How would you rate your overall stress level over the past month?",
                "options": [
                    {"value": 0, "text": "Very low"},
                    {"value": 1, "text": "Low"},
                    {"value": 2, "text": "Moderate"},
                    {"value": 3, "text": "High"},
                    {"value": 4, "text": "Very high"}
                ],
                "type": "multiple_choice"
            },
            {
                "id": "general_6",
                "text": "How would you describe your sleep quality over the past month?",
                "options": [
                    {"value": 0, "text": "Very good"},
                    {"value": 1, "text": "Good"},
                    {"value": 2, "text": "Fair"},
                    {"value": 3, "text": "Poor"},
                    {"value": 4, "text": "Very poor"}
                ],
                "type": "multiple_choice"
            },
            {
                "id": "general_7",
                "text": "Please describe how you've been feeling emotionally over the past week:",
                "type": "text_input"
            }
        ]
    
    def _get_anxiety_assessment_questions(self):
        """Get questions for the anxiety assessment."""
        return [
            {
                "id": "anxiety_1",
                "text": "Over the past 2 weeks, how often have you felt nervous, anxious, or on edge?",
                "options": [
                    {"value": 0, "text": "Not at all"},
                    {"value": 1, "text": "Several days"},
                    {"value": 2, "text": "More than half the days"},
                    {"value": 3, "text": "Nearly every day"}
                ],
                "type": "multiple_choice"
            },
            {
                "id": "anxiety_2",
                "text": "Over the past 2 weeks, how often have you been unable to stop or control worrying?",
                "options": [
                    {"value": 0, "text": "Not at all"},
                    {"value": 1, "text": "Several days"},
                    {"value": 2, "text": "More than half the days"},
                    {"value": 3, "text": "Nearly every day"}
                ],
                "type": "multiple_choice"
            },
            {
                "id": "anxiety_3",
                "text": "Over the past 2 weeks, how often have you been worrying too much about different things?",
                "options": [
                    {"value": 0, "text": "Not at all"},
                    {"value": 1, "text": "Several days"},
                    {"value": 2, "text": "More than half the days"},
                    {"value": 3, "text": "Nearly every day"}
                ],
                "type": "multiple_choice"
            },
            {
                "id": "anxiety_4",
                "text": "Over the past 2 weeks, how often have you had trouble relaxing?",
                "options": [
                    {"value": 0, "text": "Not at all"},
                    {"value": 1, "text": "Several days"},
                    {"value": 2, "text": "More than half the days"},
                    {"value": 3, "text": "Nearly every day"}
                ],
                "type": "multiple_choice"
            },
            {
                "id": "anxiety_5",
                "text": "Over the past 2 weeks, how often have you been so restless that it's hard to sit still?",
                "options": [
                    {"value": 0, "text": "Not at all"},
                    {"value": 1, "text": "Several days"},
                    {"value": 2, "text": "More than half the days"},
                    {"value": 3, "text": "Nearly every day"}
                ],
                "type": "multiple_choice"
            },
            {
                "id": "anxiety_6",
                "text": "Over the past 2 weeks, how often have you become easily annoyed or irritable?",
                "options": [
                    {"value": 0, "text": "Not at all"},
                    {"value": 1, "text": "Several days"},
                    {"value": 2, "text": "More than half the days"},
                    {"value": 3, "text": "Nearly every day"}
                ],
                "type": "multiple_choice"
            },
            {
                "id": "anxiety_7",
                "text": "Over the past 2 weeks, how often have you felt afraid as if something awful might happen?",
                "options": [
                    {"value": 0, "text": "Not at all"},
                    {"value": 1, "text": "Several days"},
                    {"value": 2, "text": "More than half the days"},
                    {"value": 3, "text": "Nearly every day"}
                ],
                "type": "multiple_choice"
            },
            {
                "id": "anxiety_8",
                "text": "Please describe any specific situations or triggers that cause you anxiety:",
                "type": "text_input"
            }
        ]
    
    def _get_depression_assessment_questions(self):
        """Get questions for the depression assessment."""
        return [
            {
                "id": "depression_1",
                "text": "Over the past 2 weeks, how often have you felt little interest or pleasure in doing things?",
                "options": [
                    {"value": 0, "text": "Not at all"},
                    {"value": 1, "text": "Several days"},
                    {"value": 2, "text": "More than half the days"},
                    {"value": 3, "text": "Nearly every day"}
                ],
                "type": "multiple_choice"
            },
            {
                "id": "depression_2",
                "text": "Over the past 2 weeks, how often have you felt down, depressed, or hopeless?",
                "options": [
                    {"value": 0, "text": "Not at all"},
                    {"value": 1, "text": "Several days"},
                    {"value": 2, "text": "More than half the days"},
                    {"value": 3, "text": "Nearly every day"}
                ],
                "type": "multiple_choice"
            },
            {
                "id": "depression_3",
                "text": "Over the past 2 weeks, how often have you had trouble falling or staying asleep, or sleeping too much?",
                "options": [
                    {"value": 0, "text": "Not at all"},
                    {"value": 1, "text": "Several days"},
                    {"value": 2, "text": "More than half the days"},
                    {"value": 3, "text": "Nearly every day"}
                ],
                "type": "multiple_choice"
            },
            {
                "id": "depression_4",
                "text": "Over the past 2 weeks, how often have you felt tired or had little energy?",
                "options": [
                    {"value": 0, "text": "Not at all"},
                    {"value": 1, "text": "Several days"},
                    {"value": 2, "text": "More than half the days"},
                    {"value": 3, "text": "Nearly every day"}
                ],
                "type": "multiple_choice"
            },
            {
                "id": "depression_5",
                "text": "Over the past 2 weeks, how often have you had poor appetite or been overeating?",
                "options": [
                    {"value": 0, "text": "Not at all"},
                    {"value": 1, "text": "Several days"},
                    {"value": 2, "text": "More than half the days"},
                    {"value": 3, "text": "Nearly every day"}
                ],
                "type": "multiple_choice"
            },
            {
                "id": "depression_6",
                "text": "Over the past 2 weeks, how often have you felt bad about yourself — or that you are a failure or have let yourself or your family down?",
                "options": [
                    {"value": 0, "text": "Not at all"},
                    {"value": 1, "text": "Several days"},
                    {"value": 2, "text": "More than half the days"},
                    {"value": 3, "text": "Nearly every day"}
                ],
                "type": "multiple_choice"
            },
            {
                "id": "depression_7",
                "text": "Over the past 2 weeks, how often have you had trouble concentrating on things, such as reading the newspaper or watching television?",
                "options": [
                    {"value": 0, "text": "Not at all"},
                    {"value": 1, "text": "Several days"},
                    {"value": 2, "text": "More than half the days"},
                    {"value": 3, "text": "Nearly every day"}
                ],
                "type": "multiple_choice"
            },
            {
                "id": "depression_8",
                "text": "Over the past 2 weeks, how often have you moved or spoken so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual?",
                "options": [
                    {"value": 0, "text": "Not at all"},
                    {"value": 1, "text": "Several days"},
                    {"value": 2, "text": "More than half the days"},
                    {"value": 3, "text": "Nearly every day"}
                ],
                "type": "multiple_choice"
            },
            {
                "id": "depression_9",
                "text": "Please describe how your mood has been affecting your daily life:",
                "type": "text_input"
            }
        ]
    
    def analyze_responses(self, assessment_type, responses):
        """
        Analyze assessment responses and generate recommendations.
        
        Args:
            assessment_type (str): Type of assessment
            responses (dict): User responses to assessment questions
            
        Returns:
            tuple: (score, recommendations, risk_level)
        """
        # Calculate score based on multiple choice questions
        score = 0
        count = 0
        
        for question_id, response in responses.items():
            if isinstance(response, (int, float)):
                score += response
                count += 1
        
        # Calculate average score if there are any numeric responses
        avg_score = score / count if count > 0 else 0
        
        # Analyze text responses using sentiment analysis
        text_responses = []
        for question_id, response in responses.items():
            if isinstance(response, str) and response.strip():
                text_responses.append(response)
        
        # Analyze text responses using sentiment analysis
        sentiment_scores = []
        for text in text_responses:
            if self.sentiment_analyzer is not None:
                try:
                    result = self.sentiment_analyzer(text)
                    # Convert sentiment to a score between 0 and 1
                    # where 0 is negative and 1 is positive
                    sentiment_score = 0.0
                    if result[0]['label'] == 'POSITIVE':
                        sentiment_score = result[0]['score']
                    else:
                        sentiment_score = 1 - result[0]['score']
                    sentiment_scores.append(sentiment_score)
                except Exception as e:
                    warnings.warn(f"Error analyzing text with transformer model: {e}. Using rule-based fallback.")
                    # Use rule-based sentiment analysis as fallback
                    result = NLPUtils._rule_based_sentiment(text)
                    if result['label'] == 'POSITIVE':
                        sentiment_scores.append(result['score'])
                    else:
                        sentiment_scores.append(1 - result['score'])
            else:
                # Use rule-based sentiment analysis
                result = NLPUtils._rule_based_sentiment(text)
                if result['label'] == 'POSITIVE':
                    sentiment_scores.append(result['score'])
                else:
                    sentiment_scores.append(1 - result['score'])
        
        # Combine numeric score and sentiment analysis
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5
        
        # Adjust score based on sentiment (sentiment is 0-1, higher is better)
        # Convert to a 0-10 scale for final score
        adjusted_score = (avg_score / 3) * 7 + (1 - avg_sentiment) * 3
        final_score = min(10, max(0, adjusted_score))
        
        # Determine risk level
        if final_score < 3:
            risk_level = "Low"
        elif final_score < 7:
            risk_level = "Moderate"
        else:
            risk_level = "High"
        
        # Generate recommendations based on assessment type and score
        recommendations = self._generate_recommendations(assessment_type, final_score, risk_level)
        
        return final_score, recommendations, risk_level
    
    def _generate_recommendations(self, assessment_type, score, risk_level):
        """
        Generate recommendations based on assessment results.
        
        Args:
            assessment_type (str): Type of assessment
            score (float): Assessment score
            risk_level (str): Risk level (Low, Moderate, High)
            
        Returns:
            list: List of recommendations
        """
        recommendations = []
        
        # General recommendations for all assessments
        recommendations.append({
            "category": "General",
            "text": "Practice self-care activities like regular exercise, healthy eating, and adequate sleep."
        })
        
        # Add recommendations based on risk level
        if risk_level == "Low":
            recommendations.append({
                "category": "Maintenance",
                "text": "Continue your current self-care practices and monitor your mental health regularly."
            })
            recommendations.append({
                "category": "Prevention",
                "text": "Consider incorporating mindfulness or meditation into your routine to maintain mental wellness."
            })
        
        elif risk_level == "Moderate":
            recommendations.append({
                "category": "Support",
                "text": "Consider talking to a trusted friend, family member, or mental health professional about your feelings."
            })
            recommendations.append({
                "category": "Self-Help",
                "text": "Explore stress reduction techniques such as deep breathing exercises, progressive muscle relaxation, or guided imagery."
            })
        
        elif risk_level == "High":
            recommendations.append({
                "category": "Professional Help",
                "text": "We strongly recommend consulting with a mental health professional for further evaluation and support."
            })
            recommendations.append({
                "category": "Immediate Support",
                "text": "If you're experiencing a crisis, please contact a mental health helpline or emergency services."
            })
        
        # Add assessment-specific recommendations
        if assessment_type == "general":
            recommendations.append({
                "category": "Resources",
                "text": "Check out the Resource Library for articles and videos on general mental wellness."
            })
        
        elif assessment_type == "anxiety":
            if risk_level == "Moderate" or risk_level == "High":
                recommendations.append({
                    "category": "Anxiety Management",
                    "text": "Practice grounding techniques when feeling anxious: focus on 5 things you can see, 4 things you can touch, 3 things you can hear, 2 things you can smell, and 1 thing you can taste."
                })
            recommendations.append({
                "category": "Resources",
                "text": "Explore anxiety management resources in our Resource Library."
            })
        
        elif assessment_type == "depression":
            if risk_level == "Moderate" or risk_level == "High":
                recommendations.append({
                    "category": "Mood Improvement",
                    "text": "Try to engage in activities you used to enjoy, even if you don't feel like it at first. Start small and be gentle with yourself."
                })
            recommendations.append({
                "category": "Resources",
                "text": "Check out depression management resources in our Resource Library."
            })
        
        return recommendations
    
    def render(self):
        """Render the self-assessment interface."""
        st.markdown("<div class='main-header'>Self-Assessment</div>", unsafe_allow_html=True)
        
        # Initialize session state for assessment
        if 'assessment_step' not in st.session_state:
            st.session_state.assessment_step = 'select'
        if 'assessment_type' not in st.session_state:
            st.session_state.assessment_type = None
        if 'assessment_responses' not in st.session_state:
            st.session_state.assessment_responses = {}
        
        # Assessment type selection
        if st.session_state.assessment_step == 'select':
            st.markdown(
                """
                <div class='info-box'>
                Please select an assessment type to begin. Each assessment focuses on different aspects of mental health.
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("<div class='sub-header'>General Assessment</div>", unsafe_allow_html=True)
                st.markdown("A broad evaluation of your overall mental well-being.")
                if st.button("Start General Assessment"):
                    st.session_state.assessment_type = "general"
                    st.session_state.assessment_step = 'questions'
                    st.session_state.assessment_responses = {}
                    st.rerun()
            
            with col2:
                st.markdown("<div class='sub-header'>Anxiety Assessment</div>", unsafe_allow_html=True)
                st.markdown("Focuses specifically on anxiety symptoms and experiences.")
                if st.button("Start Anxiety Assessment"):
                    st.session_state.assessment_type = "anxiety"
                    st.session_state.assessment_step = 'questions'
                    st.session_state.assessment_responses = {}
                    st.rerun()
            
            with col3:
                st.markdown("<div class='sub-header'>Depression Assessment</div>", unsafe_allow_html=True)
                st.markdown("Evaluates symptoms related to depression and mood disorders.")
                if st.button("Start Depression Assessment"):
                    st.session_state.assessment_type = "depression"
                    st.session_state.assessment_step = 'questions'
                    st.session_state.assessment_responses = {}
                    st.rerun()
        
        # Questionnaire
        elif st.session_state.assessment_step == 'questions':
            assessment = self.assessment_types[st.session_state.assessment_type]
            
            st.markdown(f"<div class='sub-header'>{assessment['title']}</div>", unsafe_allow_html=True)
            st.markdown(assessment['description'])
            
            st.markdown(
                """
                <div class='info-box'>
                Please answer the following questions honestly. Your responses will help us provide appropriate recommendations.
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Display questions
            with st.form("assessment_form"):
                for question in assessment['questions']:
                    if question['type'] == 'multiple_choice':
                        options = [option['text'] for option in question['options']]
                        values = [option['value'] for option in question['options']]
                        
                        index = st.radio(
                            question['text'],
                            options=range(len(options)),
                            format_func=lambda i: options[i],
                            key=f"question_{question['id']}"
                        )
                        
                        # Store the selected value
                        st.session_state.assessment_responses[question['id']] = values[index]
                    
                    elif question['type'] == 'text_input':
                        response = st.text_area(
                            question['text'],
                            key=f"question_{question['id']}"
                        )
                        
                        # Store the text response
                        st.session_state.assessment_responses[question['id']] = response
                
                submitted = st.form_submit_button("Submit Assessment")
                
                if submitted:
                    # Process the assessment
                    score, recommendations, risk_level = self.analyze_responses(
                        st.session_state.assessment_type,
                        st.session_state.assessment_responses
                    )
                    
                    # Store results in session state
                    st.session_state.assessment_results = {
                        'type': st.session_state.assessment_type,
                        'score': score,
                        'risk_level': risk_level,
                        'recommendations': recommendations,
                        'responses': st.session_state.assessment_responses,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Save to database if user is logged in
                    if st.session_state.user_name:
                        user_id = self.db.get_or_create_user(st.session_state.user_name)
                        self.db.save_assessment_result(
                            user_id,
                            st.session_state.assessment_type,
                            score,
                            st.session_state.assessment_responses,
                            recommendations
                        )
                    
                    # Move to results step
                    st.session_state.assessment_step = 'results'
                    st.rerun()
            
            # Cancel button
            if st.button("Cancel Assessment"):
                st.session_state.assessment_step = 'select'
                st.rerun()
        
        # Results
        elif st.session_state.assessment_step == 'results':
            results = st.session_state.assessment_results
            
            st.markdown(f"<div class='sub-header'>{self.assessment_types[results['type']]['title']} Results</div>", unsafe_allow_html=True)
            
            # Display score and risk level
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Score:** {results['score']:.1f}/10")
                
                # Create a gauge chart for the score
                import plotly.graph_objects as go
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = results['score'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Mental Health Score"},
                    gauge = {
                        'axis': {'range': [0, 10]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 3], 'color': "green"},
                            {'range': [3, 7], 'color': "yellow"},
                            {'range': [7, 10], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': results['score']
                        }
                    }
                ))
                
                st.plotly_chart(fig)
            
            with col2:
                st.markdown(f"**Risk Level:** {results['risk_level']}")
                
                # Display risk level explanation
                if results['risk_level'] == "Low":
                    st.markdown(
                        """
                        <div style='background-color: #d4edda; padding: 10px; border-radius: 5px;'>
                        <strong>Low Risk:</strong> Your responses suggest you're currently experiencing good mental health. 
                        Continue your self-care practices and monitor your well-being.
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                elif results['risk_level'] == "Moderate":
                    st.markdown(
                        """
                        <div style='background-color: #fff3cd; padding: 10px; border-radius: 5px;'>
                        <strong>Moderate Risk:</strong> Your responses indicate some challenges with your mental health. 
                        Consider implementing the recommendations below and reaching out for support if needed.
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                elif results['risk_level'] == "High":
                    st.markdown(
                        """
                        <div style='background-color: #f8d7da; padding: 10px; border-radius: 5px;'>
                        <strong>High Risk:</strong> Your responses suggest significant challenges with your mental health. 
                        We strongly recommend consulting with a mental health professional for support.
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            
            # Display recommendations
            st.markdown("<div class='sub-header'>Recommendations</div>", unsafe_allow_html=True)
            
            for recommendation in results['recommendations']:
                st.markdown(
                    f"""
                    <div style='background-color: #e2e3e5; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                    <strong>{recommendation['category']}:</strong> {recommendation['text']}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            # Buttons for next actions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Take Another Assessment"):
                    st.session_state.assessment_step = 'select'
                    st.rerun()
            
            with col2:
                if st.button("View Dashboard"):
                    st.session_state.current_page = 'Dashboard'
                    st.rerun()
            
            with col3:
                if st.button("Chat with Support"):
                    st.session_state.current_page = 'Chat Support'
                    st.rerun()
