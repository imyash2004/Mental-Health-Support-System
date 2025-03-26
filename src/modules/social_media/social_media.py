import streamlit as st
import pandas as pd
import plotly.express as px
from .analyzer import SocialMediaAnalyzer

class SocialMediaModule:
    """
    Social Media Module for analyzing social media posts for mental health concerns.
    """
    
    def __init__(self, db=None):
        """
        Initialize the Social Media Module.
        
        Args:
            db: Database instance (optional)
        """
        self.db = db
        self.analyzer = SocialMediaAnalyzer()
    
    def render(self):
        """Render the social media analysis interface."""
        st.markdown("<div class='main-header'>Social Media Analysis</div>", unsafe_allow_html=True)
        
        # Display information about the module
        st.markdown(
            """
            <div class='info-box'>
            This module analyzes social media posts for potential mental health concerns. 
            It can help identify early warning signs and provide appropriate recommendations.
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["Sample Analysis", "Custom Analysis", "About"])
        
        # Tab 1: Sample Analysis
        with tab1:
            st.markdown("<div class='sub-header'>Sample Social Media Posts Analysis</div>", unsafe_allow_html=True)
            
            st.markdown(
                """
                Below are sample social media posts with mental health concern analysis.
                These posts are from the provided dataset and demonstrate how the system
                can identify potential mental health concerns in social media content.
                """
            )
            
            # Get sample posts with analysis
            sample_posts = self.analyzer.get_sample_posts(5)
            
            # Display each post with analysis
            for i, post in enumerate(sample_posts):
                with st.expander(f"Post {i+1} (User ID: {post['user_id']}) - Risk Level: {post['risk_level']}", expanded=i==0):
                    st.markdown(f"**Post Content:**")
                    st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>{post['post']}</div>", unsafe_allow_html=True)
                    
                    # Display risk level with appropriate color
                    risk_color = "#d4edda"  # Green for low risk
                    if post['risk_level'] == "Moderate":
                        risk_color = "#fff3cd"  # Yellow for moderate risk
                    elif post['risk_level'] == "High":
                        risk_color = "#f8d7da"  # Red for high risk
                    
                    st.markdown(
                        f"""
                        <div style='background-color: {risk_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                        <strong>Risk Level:</strong> {post['risk_level']}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Display top concerns
                    st.markdown("**Top Concerns:**")
                    if post['top_concerns']:
                        for concern in post['top_concerns']:
                            st.markdown(f"- {concern.replace('_', ' ').title()}")
                    else:
                        st.markdown("No significant concerns detected.")
                    
                    # Display concern scores as a bar chart
                    concern_scores = post['concern_scores']
                    if any(concern_scores.values()):
                        # Convert to DataFrame for plotting
                        df = pd.DataFrame({
                            'Concern': [k.replace('_', ' ').title() for k in concern_scores.keys()],
                            'Score': list(concern_scores.values())
                        })
                        
                        # Sort by score
                        df = df.sort_values('Score', ascending=False)
                        
                        # Create bar chart
                        fig = px.bar(
                            df, 
                            x='Concern', 
                            y='Score',
                            title='Concern Scores',
                            color='Score',
                            color_continuous_scale=['green', 'yellow', 'red'],
                            range_color=[0, 1]
                        )
                        
                        st.plotly_chart(fig)
                    
                    # Display recommendations
                    st.markdown("**Recommendations:**")
                    recommendations = self.analyzer.get_recommendations(post['risk_level'], post['top_concerns'])
                    
                    for recommendation in recommendations:
                        st.markdown(
                            f"""
                            <div style='background-color: #e2e3e5; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                            <strong>{recommendation['category']}:</strong> {recommendation['text']}
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
        
        # Tab 2: Custom Analysis
        with tab2:
            st.markdown("<div class='sub-header'>Analyze Your Own Text</div>", unsafe_allow_html=True)
            
            st.markdown(
                """
                Enter any text below to analyze it for potential mental health concerns.
                This could be a social media post, a journal entry, or any other text.
                """
            )
            
            # Text input
            custom_text = st.text_area("Enter text to analyze:", height=150)
            
            if st.button("Analyze Text"):
                if custom_text.strip():
                    # Analyze the text
                    concern_scores = self.analyzer.analyze_post(custom_text)
                    risk_level = self.analyzer.get_risk_level(concern_scores)
                    
                    # Get top concerns
                    top_concerns = sorted(concern_scores.items(), key=lambda x: x[1], reverse=True)
                    top_concerns = [concern for concern, score in top_concerns if score > 0]
                    
                    # Display risk level with appropriate color
                    risk_color = "#d4edda"  # Green for low risk
                    if risk_level == "Moderate":
                        risk_color = "#fff3cd"  # Yellow for moderate risk
                    elif risk_level == "High":
                        risk_color = "#f8d7da"  # Red for high risk
                    
                    st.markdown(
                        f"""
                        <div style='background-color: {risk_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                        <strong>Risk Level:</strong> {risk_level}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Display top concerns
                    st.markdown("**Top Concerns:**")
                    if top_concerns:
                        for concern in top_concerns[:3]:  # Top 3 concerns
                            st.markdown(f"- {concern.replace('_', ' ').title()}")
                    else:
                        st.markdown("No significant concerns detected.")
                    
                    # Display concern scores as a bar chart
                    if any(concern_scores.values()):
                        # Convert to DataFrame for plotting
                        df = pd.DataFrame({
                            'Concern': [k.replace('_', ' ').title() for k in concern_scores.keys()],
                            'Score': list(concern_scores.values())
                        })
                        
                        # Sort by score
                        df = df.sort_values('Score', ascending=False)
                        
                        # Create bar chart
                        fig = px.bar(
                            df, 
                            x='Concern', 
                            y='Score',
                            title='Concern Scores',
                            color='Score',
                            color_continuous_scale=['green', 'yellow', 'red'],
                            range_color=[0, 1]
                        )
                        
                        st.plotly_chart(fig)
                    
                    # Display recommendations
                    st.markdown("**Recommendations:**")
                    recommendations = self.analyzer.get_recommendations(risk_level, top_concerns[:3])
                    
                    for recommendation in recommendations:
                        st.markdown(
                            f"""
                            <div style='background-color: #e2e3e5; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                            <strong>{recommendation['category']}:</strong> {recommendation['text']}
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                else:
                    st.warning("Please enter some text to analyze.")
        
        # Tab 3: About
        with tab3:
            st.markdown("<div class='sub-header'>About Social Media Analysis</div>", unsafe_allow_html=True)
            
            st.markdown(
                """
                ### How It Works
                
                The Social Media Analysis module uses natural language processing techniques to identify potential mental health concerns in text content. The current implementation uses a simple keyword-based approach for demonstration purposes, but a production system would use more sophisticated machine learning models.
                
                ### Key Features
                
                - **Mental Health Concern Detection**: Identifies potential mental health concerns such as depression, anxiety, suicidal ideation, and more.
                - **Risk Level Assessment**: Categorizes posts as Low, Moderate, or High risk based on the detected concerns.
                - **Personalized Recommendations**: Provides tailored recommendations based on the specific concerns identified.
                
                ### Limitations
                
                - The current implementation uses a simple keyword-based approach, which may not capture the nuances of mental health language.
                - False positives and false negatives are possible, especially with complex or ambiguous language.
                - This tool is not a replacement for professional mental health assessment and should be used as a screening tool only.
                
                ### Privacy Considerations
                
                - All analysis is performed locally and no data is sent to external servers.
                - User data is not stored unless explicitly saved by the user.
                - For a production system, proper consent and privacy measures would be implemented.
                """
            )
