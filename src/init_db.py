import os
import sys
import json
from datetime import datetime, timedelta
import random

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database.database import Database

def init_database():
    """Initialize the database with sample data."""
    print("Initializing database...")
    
    # Create database instance
    db = Database()
    
    # Add sample resources
    add_sample_resources(db)
    
    # Add sample user
    user_id = add_sample_user(db)
    
    # Add sample assessment results
    add_sample_assessment_results(db, user_id)
    
    # Add sample chat history
    add_sample_chat_history(db, user_id)
    
    print("Database initialization complete!")

def add_sample_resources(db):
    """Add sample resources to the database."""
    print("Adding sample resources...")
    
    resources = [
        {
            "title": "Mental Health Foundation",
            "description": "Information and resources for better mental health.",
            "category": "general",
            "url": "https://www.mentalhealth.org.uk/"
        },
        {
            "title": "National Alliance on Mental Illness (NAMI)",
            "description": "Mental health education, advocacy, and support.",
            "category": "general",
            "url": "https://www.nami.org/"
        },
        {
            "title": "Anxiety and Depression Association of America",
            "description": "Information, resources, and support for anxiety disorders.",
            "category": "anxiety",
            "url": "https://adaa.org/"
        },
        {
            "title": "Depression and Bipolar Support Alliance",
            "description": "Support, education, and resources for depression.",
            "category": "depression",
            "url": "https://www.dbsalliance.org/"
        },
        {
            "title": "National Suicide Prevention Lifeline",
            "description": "24/7 support for people in distress.",
            "category": "crisis",
            "url": "https://suicidepreventionlifeline.org/"
        }
    ]
    
    for resource in resources:
        db.add_resource(
            resource["title"],
            resource["description"],
            resource["category"],
            resource["url"]
        )

def add_sample_user(db):
    """Add a sample user to the database."""
    print("Adding sample user...")
    
    return db.get_or_create_user("Sample User")

def add_sample_assessment_results(db, user_id):
    """Add sample assessment results to the database."""
    print("Adding sample assessment results...")
    
    # Sample general assessment
    general_responses = {
        "general_1": 1,  # Several days
        "general_2": 1,  # Several days
        "general_3": 2,  # More than half the days
        "general_4": 1,  # Several days
        "general_5": 2,  # Moderate
        "general_6": 2,  # Fair
        "general_7": "I've been feeling a bit stressed lately due to work, but I'm managing okay. Some days are better than others."
    }
    
    general_recommendations = [
        {
            "category": "General",
            "text": "Practice self-care activities like regular exercise, healthy eating, and adequate sleep."
        },
        {
            "category": "Support",
            "text": "Consider talking to a trusted friend, family member, or mental health professional about your feelings."
        },
        {
            "category": "Self-Help",
            "text": "Explore stress reduction techniques such as deep breathing exercises, progressive muscle relaxation, or guided imagery."
        },
        {
            "category": "Resources",
            "text": "Check out the Resource Library for articles and videos on general mental wellness."
        }
    ]
    
    # Add general assessment from 30 days ago
    db.save_assessment_result(
        user_id,
        "general",
        5.2,  # Moderate risk
        general_responses,
        general_recommendations
    )
    
    # Sample anxiety assessment
    anxiety_responses = {
        "anxiety_1": 2,  # More than half the days
        "anxiety_2": 2,  # More than half the days
        "anxiety_3": 2,  # More than half the days
        "anxiety_4": 1,  # Several days
        "anxiety_5": 1,  # Several days
        "anxiety_6": 2,  # More than half the days
        "anxiety_7": 1,  # Several days
        "anxiety_8": "I feel most anxious when I have to speak in public or when I have tight deadlines at work."
    }
    
    anxiety_recommendations = [
        {
            "category": "General",
            "text": "Practice self-care activities like regular exercise, healthy eating, and adequate sleep."
        },
        {
            "category": "Support",
            "text": "Consider talking to a trusted friend, family member, or mental health professional about your feelings."
        },
        {
            "category": "Anxiety Management",
            "text": "Practice grounding techniques when feeling anxious: focus on 5 things you can see, 4 things you can touch, 3 things you can hear, 2 things you can smell, and 1 thing you can taste."
        },
        {
            "category": "Resources",
            "text": "Explore anxiety management resources in our Resource Library."
        }
    ]
    
    # Add anxiety assessment from 15 days ago
    db.save_assessment_result(
        user_id,
        "anxiety",
        6.1,  # Moderate risk
        anxiety_responses,
        anxiety_recommendations
    )
    
    # Sample depression assessment
    depression_responses = {
        "depression_1": 1,  # Several days
        "depression_2": 1,  # Several days
        "depression_3": 2,  # More than half the days
        "depression_4": 1,  # Several days
        "depression_5": 0,  # Not at all
        "depression_6": 1,  # Several days
        "depression_7": 1,  # Several days
        "depression_8": 0,  # Not at all
        "depression_9": "I've been feeling a bit down lately, but it hasn't significantly affected my daily activities."
    }
    
    depression_recommendations = [
        {
            "category": "General",
            "text": "Practice self-care activities like regular exercise, healthy eating, and adequate sleep."
        },
        {
            "category": "Support",
            "text": "Consider talking to a trusted friend, family member, or mental health professional about your feelings."
        },
        {
            "category": "Self-Help",
            "text": "Explore stress reduction techniques such as deep breathing exercises, progressive muscle relaxation, or guided imagery."
        },
        {
            "category": "Resources",
            "text": "Check out depression management resources in our Resource Library."
        }
    ]
    
    # Add depression assessment from 5 days ago
    db.save_assessment_result(
        user_id,
        "depression",
        4.5,  # Moderate risk
        depression_responses,
        depression_recommendations
    )
    
    # Add another general assessment from today
    general_responses_2 = {
        "general_1": 1,  # Several days
        "general_2": 0,  # Not at all
        "general_3": 1,  # Several days
        "general_4": 1,  # Several days
        "general_5": 1,  # Low
        "general_6": 1,  # Good
        "general_7": "I'm feeling better this week. I've been practicing some of the recommended techniques and they seem to be helping."
    }
    
    general_recommendations_2 = [
        {
            "category": "General",
            "text": "Practice self-care activities like regular exercise, healthy eating, and adequate sleep."
        },
        {
            "category": "Maintenance",
            "text": "Continue your current self-care practices and monitor your mental health regularly."
        },
        {
            "category": "Prevention",
            "text": "Consider incorporating mindfulness or meditation into your routine to maintain mental wellness."
        },
        {
            "category": "Resources",
            "text": "Check out the Resource Library for articles and videos on general mental wellness."
        }
    ]
    
    db.save_assessment_result(
        user_id,
        "general",
        3.8,  # Moderate risk (but improved)
        general_responses_2,
        general_recommendations_2
    )

def add_sample_chat_history(db, user_id):
    """Add sample chat history to the database."""
    print("Adding sample chat history...")
    
    # Sample chat conversations
    conversations = [
        {
            "user": "Hello, I'm feeling a bit anxious today.",
            "bot": "I'm sorry to hear you're feeling anxious. Would you like to talk about what's causing these feelings?",
            "sentiment": 0.3
        },
        {
            "user": "I have a big presentation at work tomorrow and I'm worried about it.",
            "bot": "It's natural to feel nervous about presentations. Would you like some tips for managing presentation anxiety?",
            "sentiment": 0.4
        },
        {
            "user": "Yes, that would be helpful.",
            "bot": "Here are some strategies that might help: 1) Practice your presentation multiple times, 2) Use deep breathing techniques before and during the presentation, 3) Remember that it's okay to pause and collect your thoughts, 4) Focus on your message rather than on yourself. Would you like to explore any of these strategies further?",
            "sentiment": 0.6
        },
        {
            "user": "I'll try the deep breathing techniques. How do I do that?",
            "bot": "Deep breathing can be very effective for reducing anxiety. Try this simple technique: 1) Inhale slowly through your nose for a count of 4, 2) Hold your breath for a count of 2, 3) Exhale slowly through your mouth for a count of 6, 4) Repeat this cycle several times. Would you like to try this now?",
            "sentiment": 0.7
        },
        {
            "user": "I just tried it and it does help a bit. Thank you.",
            "bot": "You're welcome! I'm glad it helped. Remember you can use this technique anytime you feel anxious, including right before your presentation. Is there anything else you'd like support with?",
            "sentiment": 0.8
        },
        {
            "user": "I've also been having trouble sleeping lately because of stress.",
            "bot": "Sleep difficulties can be challenging, especially when you're under stress. Would you like some suggestions for improving your sleep quality?",
            "sentiment": 0.4
        },
        {
            "user": "Yes, please.",
            "bot": "Here are some strategies that might help improve your sleep: 1) Establish a regular sleep schedule, 2) Create a relaxing bedtime routine, 3) Limit screen time before bed, 4) Create a comfortable sleep environment, 5) Avoid caffeine and large meals before bedtime. Would you like more information about any of these strategies?",
            "sentiment": 0.6
        },
        {
            "user": "Those are helpful. I'll try to implement them. Thanks for your help.",
            "bot": "You're very welcome. Remember, it's okay to reach out for support when you need it. Would you like to take a self-assessment to better understand your current mental health status?",
            "sentiment": 0.7
        }
    ]
    
    # Add conversations to database
    for conversation in conversations:
        db.save_chat_message(
            user_id,
            conversation["user"],
            conversation["bot"],
            conversation["sentiment"]
        )

if __name__ == "__main__":
    init_database()
