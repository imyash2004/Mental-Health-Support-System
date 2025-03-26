import streamlit as st
import pandas as pd
import numpy as np
from modules.self_assessment.assessment import SelfAssessmentModule
from modules.chatbot.chatbot import ChatbotModule
from modules.dashboard.dashboard import DashboardModule
from modules.social_media.social_media import SocialMediaModule
from modules.database.database import Database

# Set page configuration
st.set_page_config(
    page_title="Mental Health Support System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database (with error handling)
@st.cache_resource
def init_database():
    try:
        return Database()
    except Exception as e:
        st.warning(f"Database initialization failed: {e}. Using mock data instead.")
        return None

db = init_database()

# Custom CSS
def load_css():
    st.markdown("""
    <style>
        /* Main theme colors */
        :root {
            --primary-color: #6a98f0;
            --secondary-color: #8a4fff;
            --accent-color: #38b6ff;
            --background-color: #1e1e2e;
            --card-background: #2a2a3a;
            --text-color: #ffffff;
            --muted-text: #b0b0c0;
            --success-color: #4cd97b;
            --warning-color: #ffcc5c;
            --danger-color: #ff5c5c;
            --info-color: #5ce1ff;
        }
        
        /* Override Streamlit's default styles */
        .stApp {
            background-color: var(--background-color);
        }
        
        /* Headers */
        .main-header {
            font-size: 2.5rem;
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
            font-weight: bold;
        }
        
        .sub-header {
            font-size: 1.5rem;
            color: var(--accent-color);
            margin-bottom: 1rem;
            font-weight: 600;
        }
        
        /* Boxes */
        .info-box {
            background-color: var(--card-background);
            border-left: 4px solid var(--info-color);
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            color: var(--text-color);
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        }
        
        /* Buttons */
        .stButton>button {
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: 0.5rem;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: var(--secondary-color);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        
        /* Emergency button */
        .emergency-button {
            background-color: var(--danger-color);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            text-align: center;
            margin: 1rem 0;
            cursor: pointer;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }
        
        .emergency-button:hover {
            background-color: #ff3333;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }
        
        /* Chat messages */
        .user-message {
            background-color: var(--primary-color);
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            text-align: right;
            color: white;
        }
        
        .bot-message {
            background-color: var(--card-background);
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            border-left: 3px solid var(--accent-color);
            color: var(--text-color);
        }
        
        /* Form inputs */
        .stTextInput>div>div>input {
            background-color: var(--card-background);
            color: var(--text-color);
            border: 1px solid var(--primary-color);
        }
        
        .stTextArea>div>div>textarea {
            background-color: var(--card-background);
            color: var(--text-color);
            border: 1px solid var(--primary-color);
        }
    </style>
    """, unsafe_allow_html=True)

load_css()

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'
if 'user_name' not in st.session_state:
    st.session_state.user_name = ''
if 'assessment_results' not in st.session_state:
    st.session_state.assessment_results = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar navigation
st.sidebar.markdown("<div class='main-header'>Mental Health Support</div>", unsafe_allow_html=True)

# User information section
with st.sidebar.expander("User Information", expanded=True):
    if st.session_state.user_name:
        st.write(f"Welcome, {st.session_state.user_name}!")
    else:
        user_name = st.text_input("Your Name")
        if user_name:
            st.session_state.user_name = user_name
            st.rerun()

# Navigation
st.sidebar.markdown("<div class='sub-header'>Navigation</div>", unsafe_allow_html=True)
pages = {
    'Home': 'Home',
    'Self Assessment': 'Take a Self-Assessment',
    'Chat Support': 'Chat with Support Bot',
    'Dashboard': 'Your Dashboard',
    'Social Media': 'Social Media Analysis',
    'Resources': 'Resource Library'
}

for page, label in pages.items():
    if st.sidebar.button(label, key=page):
        st.session_state.current_page = page
        st.rerun()

# Emergency contact section
st.sidebar.markdown("<div class='sub-header'>Emergency Support</div>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<div class='emergency-button'>🆘 Emergency Contact</div>",
    unsafe_allow_html=True
)
st.sidebar.markdown(
    "If you're in crisis, please call the Mental Health Helpline: **1-800-123-4567**"
)

# Main content area
def render_home():
    st.markdown("<div class='main-header'>Welcome to the Mental Health Support System</div>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class='info-box'>
        This platform provides tools and resources to support your mental health journey. 
        Here's what you can do:
        
        - Take a self-assessment to understand your mental health status
        - Chat with our support bot for guidance and resources
        - Track your progress through the dashboard
        - Access a library of mental health resources
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='sub-header'>Get Started</div>", unsafe_allow_html=True)
        st.markdown(
            """
            If you're new here, we recommend starting with a self-assessment.
            This will help us understand how we can best support you.
            """
        )
        if st.button("Take Self-Assessment"):
            st.session_state.current_page = 'Self Assessment'
            st.rerun()
    
    with col2:
        st.markdown("<div class='sub-header'>Need to Talk?</div>", unsafe_allow_html=True)
        st.markdown(
            """
            Our AI chatbot is here to listen and provide support.
            While not a replacement for professional help, it can offer guidance and resources.
            """
        )
        if st.button("Chat Now"):
            st.session_state.current_page = 'Chat Support'
            st.rerun()

# Initialize modules with error handling
try:
    self_assessment_module = SelfAssessmentModule(db)
except Exception as e:
    st.warning(f"Self Assessment Module initialization failed: {e}")
    self_assessment_module = None

try:
    chatbot_module = ChatbotModule(db)
except Exception as e:
    st.warning(f"Chatbot Module initialization failed: {e}")
    chatbot_module = None

try:
    dashboard_module = DashboardModule(db)
except Exception as e:
    st.warning(f"Dashboard Module initialization failed: {e}")
    dashboard_module = None

try:
    social_media_module = SocialMediaModule(db)
except Exception as e:
    st.warning(f"Social Media Module initialization failed: {e}")
    social_media_module = None

# Render the appropriate page with error handling
if st.session_state.current_page == 'Home':
    render_home()
elif st.session_state.current_page == 'Self Assessment':
    if self_assessment_module:
        self_assessment_module.render()
    else:
        st.error("Self Assessment Module is not available. Please check the console for errors.")
elif st.session_state.current_page == 'Chat Support':
    if chatbot_module:
        chatbot_module.render()
    else:
        st.error("Chatbot Module is not available. Please check the console for errors.")
elif st.session_state.current_page == 'Dashboard':
    if dashboard_module:
        dashboard_module.render()
    else:
        st.error("Dashboard Module is not available. Please check the console for errors.")
elif st.session_state.current_page == 'Social Media':
    if social_media_module:
        social_media_module.render()
    else:
        st.error("Social Media Module is not available. Please check the console for errors.")
elif st.session_state.current_page == 'Resources':
    st.markdown("<div class='main-header'>Resource Library</div>", unsafe_allow_html=True)
    st.markdown("This section will contain mental health resources and educational materials.")
    st.info("Resource Library is under development.")
